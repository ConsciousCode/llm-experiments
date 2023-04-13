#!/usr/bin/env python3
'''
Transformer library for PyTorch. Divides transformers into an unusual split for
maximum flexibility:

Transformer - residuals
	Attention - projections and heads
		Selector - query-key selector of values, eg softmax(Q K^T) V
	Feedforward

InfoStill - Transformer that adds feedback to every layer

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, TypeVar, TypeAlias, Callable, Mapping, Iterator, Type
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from abc import abstractmethod

# Defaults
NORM = 1e-5
DROP = 0.1
BIAS = True

# Helper functions

T = TypeVar("T")
def default(x: Optional[T], y: T|Callable[[], T]) -> T:
	'''Extensible defaults for function arguments.'''
	return x if x is not None else y() if callable(y) else y

ConfigMod: TypeAlias = Optional[bool|float|nn.Module]

def mod_or_config(config: ConfigMod, default: float, mod: Type[nn.Module]):
	'''Returns a module or a module initialized with a config.'''
	if isinstance(config, float):
		return mod(config) if config else None
	if isinstance(config, bool):
		return mod(default) if config else None
	return config

def output_append(ctx, key, value):
	'''Appends to the context output if it exists'''
	out = ctx['output'].get(key, None)
	if out is not None:
		out.append(value)

class RotaryEmbedding(nn.Module):
	'''
	Rotary Embedding (RoPE)
	
	RoFormer: Enhanced Transformer with Rotary Position Embedding
	https://arxiv.org/abs/2104.09864
	'''
	
	def __init__(self, dim, base=10000.0):
		super().__init__()
		self.dim = dim
		self.cache = {}
		inv_freq = base ** -(torch.arange(0, dim, 2).float() / dim)
		emb = torch.empty(self.dim)
		emb[::2] = inv_freq
		emb[1::2] = inv_freq
		self.register_buffer('inv_freq', emb)

	def forward(self, q, k):
		assert q.shape == k.shape
		seq = q.shape[1]
		
		sc = self.cache.get(seq, None)
		if sc is None:
			sin, cos = torch.sin(seq * self.inv_freq), torch.cos(seq * self.inv_freq)
			self.cache[seq] = sin, cos
		else:
			sin, cos = sc

		def rotate(x):
			# Rotate half
			x1, x2 = x[..., :-1:2], x[..., 1::2]
			x_rot = torch.cat((-x2, x1), dim=-1)
			return (x * cos) + (x_rot * sin)
		
		return rotate(q), rotate(k)

class ScaledDotProductSelector(nn.Module):
	'''
	Traditional softmax(Q K^T) V attention selector.
	'''
	
	def __init__(self, max_seq: Optional[int], dropout: ConfigMod=DROP):
		'''
		Parameters:
			max_seq: Maximum sequence length
			dropout: Attention dropout
		'''
		super().__init__()
		
		self.attn_dropout = mod_or_config(dropout, DROP, nn.Dropout)
		
		self.register_buffer("bias",
			torch.tril(torch.ones((max_seq, max_seq), dtype=torch.bool)).view(
				1, 1, max_seq, max_seq
			) if max_seq is not None else None
		)
	
	def forward(self, q, k, v, **ctx):
		attn_weight = torch.matmul(q, k.transpose(-1, -2))
		
		dtype, device = attn_weight.dtype, attn_weight.device
		qlen, klen = q.shape[-2], k.shape[-2]
		
		# Causal mask
		if self.bias is not None:
			causal_mask = self.bias[:, :, klen - qlen : klen, :klen]
			mask_value = torch.finfo(dtype).min
			mask_value = torch.full([], mask_value, dtype=dtype).to(device)
			attn_weight = torch.where(causal_mask, attn_weight, mask_value)
			
			attn_mask = ctx.get('attn_mask', None)
			if attn_mask is not None:
				attn_weight = attn_weight + attn_mask
		
		attn_weight = F.softmax(attn_weight, dim=-1)
		
		# Downcast (if necessary) back to V's dtype
		if self.attn_dropout is not None:
			attn_weight = self.attn_dropout(attn_weight.type(v.dtype))
		
		head_mask = ctx.get('head_mask', None)
		if head_mask is not None:
			attn_weight = attn_weight * head_mask
		
		attn = torch.matmul(attn_weight, v)
		
		output_append(ctx, 'attention', attn_weight)
		
		return attn

class Attention(nn.Module):
	'''
	Base class for all attention layers. Mostly the norms and dropouts. Some
	forms of attention don't even have the QKV projections.
	
	Attention Is All You Need
	https://arxiv.org/abs/1706.03762
	'''
	
	def __init__(self,
			embed: int,
			*,
			prenorm: ConfigMod=NORM,
			dropout: ConfigMod=DROP,
			postnorm: ConfigMod=None,
			
			bias=BIAS
		):
		'''
		Parameters:
			embed: Embedding dimension
			max_seq: Maximum sequence length
			
			prenorm: Whether to use pre-normalization
			dropout: Residual dropout
			postnorm: Whether to use post-normalization
			
			bias: Whether to use bias
		'''
		super().__init__()
		self.embed = embed
		
		self.prenorm = mod_or_config(prenorm, NORM, nn.LayerNorm)
		self.resid_dropout = mod_or_config(dropout, DROP, nn.Dropout)
		self.out_proj = nn.Linear(embed, embed, bias)
		self.postnorm = mod_or_config(postnorm, NORM, nn.LayerNorm)
	
	@abstractmethod
	def _attention(self, x, **ctx) -> torch.Tensor:
		'''
		Perform attention-specific operations to convert input to Q, K, and V,
		then into an attention output.
		'''
		pass
	
	def forward(self, x, **ctx):
		'''
		Normalizations, projections, and dropouts.
		'''
		if self.prenorm is not None:
			x = self.prenorm(x)
		
		atn = self._attention(x, **ctx)
		atn = self.out_proj(atn)
		
		if self.resid_dropout is not None:
			atn = self.resid_dropout(atn)
		
		if self.postnorm is not None:
			atn = self.postnorm(atn)
		
		output_append(ctx, 'hidden', atn)
		
		return atn

class MultiheadAttention(Attention):
	'''
	Normal mutli-headed attention with qkv and output projections.
	'''
	
	def __init__(self,
			embed: int,
			heads: int=1,
			*,
			max_seq: Optional[int]=None,
			rotary_embed: Optional[bool|nn.Module]=None,
			selector: Optional[nn.Module]=None,
			
			prenorm: ConfigMod=NORM,
			dropout: ConfigMod=DROP,
			postnorm: ConfigMod=None,
			
			qknorm: bool|float=True,
			bias=BIAS
		):
		'''
		Parameters:
			embed: Embedding dimension
			max_seq: Maximum sequence length
			
			rotary_embed: Whether to use rotary embedding, or the embedding module
			selector: selector module
			
			prenorm: Whether to use pre-normalization
			dropout: Residual dropout
			postnorm: Whether to use post-normalization
			
			qknorm: Whether to normalize queries and keys
			bias: Whether to use bias
		'''
		super().__init__(
			embed,
			prenorm=prenorm,
			dropout=dropout,
			postnorm=postnorm,
			bias=bias
		)
		self.heads = heads
		
		if isinstance(rotary_embed, bool):
			self.rotary_embed = RotaryEmbedding(embed) if rotary_embed else None
		else:
			self.rotary_embed = rotary_embed
		
		if selector is None:
			assert max_seq is not None, "max_seq must be specified if selector is not"
			selector = ScaledDotProductSelector(max_seq)
		self.selector = selector
		
		if qknorm: # True or float
			if qknorm is True:
				qknorm = embed ** -0.5
			self.qknorm = nn.Parameter(torch.tensor(qknorm))
		else: # None or False
			self.qknorm = None
		
		# Original GPT-2 uses Conv1D from pytorch_util to take advantage of addmm
		#  which is marginally faster than matmul. This isn't necessary for newer
		#  versions of Torch which automatically use admm for 2D matmul.
		
		# Cramming: Training a Language Model on a Single GPU in One Day
		# https://arxiv.org/abs/2212.14034
		# * Removing bias has negligible effect on loss and reduces parameters
		self.qkv_proj = nn.Linear(embed, embed * 3, bias)
	
	def _split_heads(self, x):
		x = x.view(*x.shape[:-1], self.heads, -1)
		return x.transpose(-2, -3)
	
	def _merge_heads(self, x):
		x = x.transpose(-2, -3)
		return x.reshape(*x.shape[:-2], self.embed)
	
	def _attention(self, x, **ctx):
		q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
		
		if self.rotary_embed is not None:
			q, k = self.rotary_embed(q, k)
		
		q, k, v = map(self._split_heads, (q, k, v))
		
		# Query-Key Normalization for Transformers
		# https://arxiv.org/abs/2010.04245
		if self.qknorm is not None:
			q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
			q, k = q * self.qknorm, k * self.qknorm
		
		# Caching for faster inference
		output = ctx['output']
		cache = output.get('cache', None)
		if cache is not None:
			past_k, past_v = cache
			k = torch.cat((past_k, k), dim=-2)
			v = torch.cat((past_v, v), dim=-2)
			output['cache'] = (k, v)
		
		x = self.selector(q, k, v, **ctx)
		return self._merge_heads(x)

class DynamicMemoryQueryAndUpdate(torch.autograd.Function):
	'''
	Core autograd function for DynamicMemorySelector.
	'''
	
	@staticmethod
	def forward(ctx, q, k, v, memory):
		mk, mv = memory.search(q)
		memory.update(k, v)
		return mk, mv

	@staticmethod
	def backward(ctx, mk_grad, mv_grad):
		# STE
		# K'/V' = query, K/V = update
		# Q ~= K'/K + noise
		# K'/V' ~= K/V
		return mk_grad, mk_grad, mv_grad, None

class DynamicMemorySelector(nn.Module):
	'''
	Discretized external memory query-and-update operation. The external memory
	should approximate attention over an arbitrarily large external memory vector.
	Useful for personalization, facts, and memory-heavy tasks.
	
	1. Query a number of entries from the memory
	2. Use the gate to select keys and values and the add/erase operation
	3. Update the memory with the new keys and values
	4. Return the original query
	
	Memorizing Transformers
	https://arxiv.org/abs/2203.08913
	* kNN memory, paper uses it as an alternative to recurrency
	
	Neural Turing Machines
	https://arxiv.org/abs/1410.5401
	* Read and write/erase heads, paper uses it for memory-augmented tasks
	'''
	
	def __init__(self, selector: Optional[nn.Module]=None):
		'''
		Parameters:
			selector: selector module after memory lookup
		'''
		super().__init__()
		self.selector = default(selector, lambda: ScaledDotProductSelector())
	
	def forward(self, q, k, v, **ctx):
		output_append(ctx, 'attention', None)
		k, v = DynamicMemoryQueryAndUpdate.apply(q, k, v, ctx['dynamic_memory'])
		return self.selector(q, k, v, **ctx)

class StaticMemoryQuery(torch.autograd.Function):
	'''
	Core autograd function for StaticMemorySelector.
	'''
	@staticmethod
	def forward(ctx, q, memory):
		# Approximate Nearest Neighbor
		ann = memory.search(q)

		# Compute the dot product similarity and softmax
		sim = torch.einsum('bsd,bskd->bsk', q, ann)
		weight = torch.softmax(sim, dim=-1)

		# Compute the weighted sum of the nearest neighbors
		wsum = torch.einsum('bsk,bskd->bsd', weight, ann)
		
		ctx.save_for_backward(ann, weight)
		
		return wsum
	
	@staticmethod
	def backward(ctx, grad_wsum):
		ann, weight = ctx.saved_tensors
		
		grad_weight = torch.einsum('bsd,bskd->bsk', grad_wsum, ann)
		grad_weight = grad_weight.unsqueeze(-1).unsqueeze(-1)
		
		eye = torch.eye(ann.shape[-1], device=ann.device).unsqueeze(0).unsqueeze(0)
		grad_sim = torch.einsum('bsij,bski,bskj->bsk',
			grad_weight, eye - weight.unsqueeze(-1), weight.unsqueeze(-2)
		)
		grad_q = torch.einsum('bsk,bskd->bsd', grad_sim, ann)
		
		return grad_q, None

class StaticMemoryAttention(Attention):
	'''
	Discretized memory lookup. Finds the top-1 embeddings in the memory and
	returns those embeddings. Useful for document-heavy memorization. This is
	inspired by the current usage of embedding to document databases like
	Pinecone and Milvus.
	
	Replaces feedforward layers.
	
	Attention recontextualized as kNN memory. Queries, inserts keys and
	values, and returns the results of the query.
	
		Transformer Feed-Forward Layers Are Key-Value Memories
		https://arxiv.org/abs/2012.14913
		
		Augmenting Self-attention with Persistent Memory
		https://arxiv.org/pdf/1907.01470.pdf
		* Proves FF networks are equivalent to attention with static memory
		
		Attention Approximates Sparse Distributed Memory
		https://arxiv.org/abs/2111.05498
		* Theoretical basis for why FF might be both attention and memory
	'''
	
	def __init__(self,
			embed: int,
			*,
			prenorm: ConfigMod=NORM,
			dropout: ConfigMod=DROP,
			postnorm: ConfigMod=None,
			
			bias=BIAS
		):
		'''
		Parameters:
			embed: Embedding dimension
			
			prenorm: Whether to use pre-normalization
			dropout: Residual dropout
			postnorm: Whether to use post-normalization
			
			bias: Whether to use bias
		'''
		super().__init__(
			embed,
			prenorm=prenorm,
			dropout=dropout,
			postnorm=postnorm,
			bias=bias
		)
		self.query_proj = nn.Linear(embed, embed, bias)
	
	def _attention(self, x, **ctx):
		output_append(ctx, 'attention', None)
		q = self.query_proj(x)
		return StaticMemoryQuery.apply(q, ctx['static_memory'])

class Residual(nn.ModuleList):
	'''
	Adds residuals to each sequential model.
	'''
	
	def forward(self, x, **ctx):
		for layer in self:
			x = x + layer(x, **ctx)
		return x

class Instill(Residual):
	'''
	INformation Still. Information goes up and comes back down condensed and
	distilled. This is a generalization of the residual connection.
	
	Adds residual of upper layers to lower layers as feedback.
	
	Addressing Some Limitations of Transformers with Feedback Memory
	https://arxiv.org/abs/2002.09402
	* Using output of upper layers for lower (modified: per layer pair, no pooling)
	
	Memory transformers
	https://arxiv.org/abs/2006.11527
	* Concatenating memory to input tokens (modified: no memory controller)
	'''
	
	def _feedback(self, x, f, **ctx):
		'''How to apply the feedback'''
		return x if f is None else x + f
	
	def forward(self, x, **ctx):
		f = ctx.get('feedback', None)
		if f is None:
			f = [None] * len(self)
		
		for i, layer in enumerate(self):
			print("Layer", i)
			x = x + layer(self._feedback(x, f[i], **ctx), **ctx)
		return x

class LanguageModel(nn.Module):
	'''
	Wraps a language model with a token embedding and a linear output layer.
	'''
	def __init__(self,
			vocab: int,
			embed: int,
			model: list[nn.Module],
			dropout: Optional[nn.Module]=None,
			postnorm: Optional[nn.Module]=None,
			dtype: Optional[torch.dtype]=None
		):
		'''
		Parameters
			vocab: Vocabulary size
			embed: Embedding size
			model: Language model
			dropout: Embedding dropout layer
			postnorm: Post-normalization layer
			dtype: Data type
		'''
		super().__init__()
		
		self.embed = nn.Embedding(vocab, embed)
		self.embed_dropout = dropout
		self.lm = model
		self.postnorm = postnorm
		self.lm_head = nn.Linear(vocab, embed, bias=BIAS)
		self.lm_head.weight = self.embed.weight
		self.dtype = default(dtype, torch.float32)
	
	def forward(self,
			x: torch.Tensor,
			*,
			static_memory: Optional[object]=None,
			dynamic_memory: Optional[object]=None,
			feedback: Optional[list[torch.Tensor]]=None,
			cache: Optional[tuple[torch.Tensor, torch.Tensor]]=None,
			
			# Masks
			attention_mask: Optional[torch.Tensor]=None,
			head_mask: Optional[torch.Tensor]=None,
			
			# Flags
			output_attention=False,
			output_hidden=False
		) -> tuple:
		'''
		kwargs:
			x: Input embeddings (NOT ids)
			feedback: Feedback from previous timestep
			cache: key and value for past attention
			
			attention_mask: Mask for attention
			head_mask: Mask for attention heads
			
			output_attention: Output attention
			output_hidden: Output hidden states
			use_cache: Use cache for attention
		'''
		
		# Convert ids to embedding
		if not x.is_floating_point():
			x = self.embed(x)
		
		# GPT2Attention mask
		if attention_mask is not None:
			attention_mask = attention_mask.view(x.shape[0], -1)
			attention_mask = attention_mask[:, None, None, :]
			
			# Adjust to (-inf, 0] for additive mask
			attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
			attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min
		
		if self.embed_dropout is not None:
			x = self.embed_dropout(x)
		
		output = {
			"cache": [] if cache is not None else None,
			"hidden": [] if output_hidden else None,
			"attention": [] if output_attention else None
		}
		
		self.lm(x,
		 	dynamic_memory=dynamic_memory,
			static_memory=static_memory,
			feedback=feedback,
			cache=cache,
			
			attention_mask=attention_mask,
			head_mask=head_mask,
			
		 	output=output
		)
		
		if output_hidden:
			output['hidden'].append(x)
		
		# Process final hidden state
		if self.postnorm is not None:
			x = self.postnorm(x)
		
		return x, output

class MyModelBlock(nn.Module):
	def __init__(self, attn, smem):
		super().__init__()
		self.attn = attn
		self.smem = smem
	
	def forward(self, x, **ctx):
		x = self.attn(x, **ctx)
		x = self.smem(x, **ctx)
		return x

class MyModel(Instill):
	'''
	Model designed to load GPT-2 weights
	'''
	
	def __init__(self,
			embed: int,
			layers: int,
			heads: int,
			max_seq: int,
			
			# Norms
			norm_epsilon: Optional[float]=NORM,
			prenorm_epsilon: Optional[float]=None,
			postnorm_epsilon: Optional[float]=None,
			
			# Dropout
			dropout_p: Optional[float]=DROP,
			pdrop_residual: Optional[float]=None,
			pdrop_attn: Optional[float]=None
		):
		'''
		Parameters:
			embed - Embedding size
			layers - Number of layers
			heads - Number of heads
			max_seq - Maximum sequence length
			
			norm_epsilon - Epsilon default for norms, defaults to NORM
			prenorm_epsilon - Epsilon for transformer prenorms
			postnorm_epsilon - Epsilon for model's postnorm
			
			dropout_p - Default dropout rate, defaults to DROP
			pdrop_residual - Residual dropout rate
			pdrop_attn - Attention dropout rate
		'''
		
		prenorm_epsilon = default(prenorm_epsilon, norm_epsilon)
		postnorm_epsilon = default(postnorm_epsilon, norm_epsilon)
		
		pdrop_residual = default(pdrop_residual, dropout_p) or None
		pdrop_attn = default(pdrop_attn, dropout_p) or None
		
		def prenorm_dropout():
			return {
				"prenorm": nn.LayerNorm(embed, prenorm_epsilon),
				"dropout": nn.Dropout(pdrop_residual)
			}
		
		# Note to self: if we want this to succeed as a proof of concept, we
		#  can't be too experimental with the setup. It needs to be roughly
		#  1:1 with the teacher model, even if that doesn't make complete
		#  sense. Here, MHA/SMA relates directly to GPT-2 MHA/MLP
		def block(rot=False):
			return MyModelBlock(
				MultiheadAttention(embed, heads,
					rotary_embed=rot,
					selector=ScaledDotProductSelector(
						max_seq, dropout=pdrop_attn
					),
					**prenorm_dropout()
				),
				StaticMemoryAttention(embed, **prenorm_dropout())
			)
		
		super().__init__([
			block(True), *(block() for i in range(1, layers))
		])

class MyGPT2Model(MyModel):
	'''
	MyModel configured using GPT-2 parameters.
	'''
	
	def __init__(self, config: GPT2Config):
		'''
		Relevant config parameters:
			n_positions
				The maximum sequence length that this model might ever be used with.
			n_embd
				Dimensionality of the embeddings and hidden states.
			n_layer
				Number of hidden layers in the Transformer encoder.
			n_head
				Number of attention heads for each attention layer in the Transformer encoder.
			resid_pdrop
				The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
			embd_pdrop
				The dropout ratio for the embeddings.
			attn_pdrop
				The dropout ratio for the attention.
			layer_norm_epsilon
				The epsilon to use in the layer normalization layers.
		'''
		super().__init__(
			embed=config.n_embd,
			layers=config.n_layer,
			heads=config.n_head,
			max_seq=config.n_positions,
			
			dropout_p=config.resid_pdrop,
			pdrop_residual=config.resid_pdrop,
			pdrop_attn=config.attn_pdrop,
			
			prenorm_epsilon=config.layer_norm_epsilon,
			postnorm_epsilon=config.layer_norm_epsilon
		)

class TransformersWrapper(nn.Module):
	'''
	Wrapper converting transformers-style inputs to InfoStill-style inputs.
	'''
	
	def __init__(self, model):
		super().__init__()
		
		self.model = model
	
	def forward(self, *, input_ids, **kwargs):
		return self.model(input_ids, **kwargs)