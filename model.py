#!/usr/bin/env python3

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
			dropout: ConfigMod=DROP
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
		
		self.resid_dropout = mod_or_config(dropout, DROP, nn.Dropout)
		self.c_proj = nn.Linear(embed, embed)
	
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
		atn = self._attention(x, **ctx)
		atn = self.c_proj(atn)
		
		if self.resid_dropout is not None:
			atn = self.resid_dropout(atn)
		
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
			selector: Optional[nn.Module]=None,
			
			dropout: ConfigMod=DROP
		):
		'''
		Parameters:
			embed: Embedding dimension
			max_seq: Maximum sequence length
			
			selector: selector module
			
			prenorm: Whether to use pre-normalization
			dropout: Residual dropout
			postnorm: Whether to use post-normalization
			
			qknorm: Whether to normalize queries and keys
			bias: Whether to use bias
		'''
		super().__init__(
			embed,
			dropout=dropout
		)
		self.heads = heads
		
		if selector is None:
			assert max_seq is not None, "max_seq must be specified if selector is not"
			selector = ScaledDotProductSelector(max_seq)
		self.selector = selector
		
		self.scale = embed ** -0.25
		
		# Original GPT-2 uses Conv1D from pytorch_util to take advantage of addmm
		#  which is marginally faster than matmul. This isn't necessary for newer
		#  versions of Torch which automatically use admm for 2D matmul.
		
		# Cramming: Training a Language Model on a Single GPU in One Day
		# https://arxiv.org/abs/2212.14034
		# * Removing bias has negligible effect on loss and reduces parameters
		self.c_attn = nn.Linear(embed, embed * 3)
	
	def _split_heads(self, x):
		x = x.view(*x.shape[:-1], self.heads, -1)
		return x.transpose(-2, -3)
	
	def _merge_heads(self, x):
		x = x.transpose(-2, -3)
		return x.reshape(*x.shape[:-2], self.embed)
	
	def _attention(self, x, **ctx):
		q, k, v = self.c_attn(x).chunk(3, dim=-1)
		q, k, v = map(self._split_heads, (q, k, v))
		q, k = q * self.scale, k * self.scale
		x = self.selector(q, k, v, **ctx)
		return self._merge_heads(x)

class DynamicMemoryQueryAndUpdate(torch.autograd.Function):
	'''
	Core autograd function for DynamicMemorySelector.
	'''
	
	@staticmethod
	def forward(ctx, q, k, v, memory):
		# TODO: Assumes k=1, no weighted sum
		mk, mv = memory.search(q)
		memory.update(k, v)
		return mk, mv

	@staticmethod
	def backward(ctx, mk_grad, mv_grad):
		return mk_grad, mk_grad, mv_grad, None

class DynamicMemoryAttention(Attention):
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
	
	def __init__(self, embed, selector: Optional[nn.Module]=None, dropout=DROP, bias=BIAS):
		'''
		Parameters:
			selector: selector module after memory lookup
		'''
		
		super().__init__(
			embed,
			dropout=dropout,
			bias=bias
		)
		self.qkv_proj = nn.Linear(embed, embed, bias)
		self.selector = default(selector, lambda: ScaledDotProductSelector())
	
	def _attention(self, x, **ctx):
		output_append(ctx, 'attention', None)
		q, k, v = self.qkv_proj(x)
		k, v = DynamicMemoryQueryAndUpdate.apply(q, k, v, ctx['static_memory'])
		return self.selector(q, k, v, **ctx)

class StaticMemoryQuery(torch.autograd.Function):
	'''
	Core autograd function for StaticMemoryAttention.
	'''
	
	@staticmethod
	def forward(ctx, q, memory):
		# Approximate Nearest Neighbor
		ann = memory.search(q).to(q.device)

		# Compute the dot product similarity and softmax
		sim = torch.einsum('bsd,bskd->bsk', q, ann)
		weight = torch.softmax(sim, dim=-1)

		# Compute the weighted sum of the nearest neighbors
		wsum = torch.einsum('bsk,bskd->bsd', weight, ann)
		
		ctx.save_for_backward(ann, weight)
		
		return wsum
	
	@staticmethod
	def backward(ctx, grad_wsum):
		# Not 100% sure this is the right gradient for softmax and weighted sum
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
			dropout: ConfigMod=DROP,
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
			dropout=dropout
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

class LanguageModel(nn.Module):
	'''
	Wraps a language model with a token embedding and a linear output layer.
	'''
	
	def __init__(self,
			vocab: int,
			embed: int,
			max_seq: int,
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
		
		self.wte = nn.Embedding(vocab, embed)
		self.wpe = nn.Embedding(max_seq, embed)
		self.drop = dropout
		self.transformer = model
		self.ln_f = postnorm
		# Tie lm_head and embed weights
		self.lm_head = nn.Linear(vocab, embed, bias=False)
		self.lm_head.weight = self.wte.weight
		self.dtype = default(dtype, torch.float32)
	
	def embed(self, x, positional=True):
		x = self.wte(x)
		if positional:
			pos = torch.arange(0, x.shape[-1], device=x.device)
			pos = pos.unsqueeze(0).view(-1, x.shape[-1])
			print("EMBED", x.shape, pos.shape)
			x = x + self.wpe(pos)
		return x
	
	def forward(self,
			x: torch.Tensor,
			*,
			static_memory: Optional[object]=None,
			dynamic_memory: Optional[object]=None,
			
			# Masks
			attention_mask: Optional[torch.Tensor]=None,
			
			# Flags
			output_attention=False,
			output_hidden=False
		) -> tuple:
		'''
		kwargs:
			x: Input embeddings or ids
			static_memory: Static memory
			dynamic_memory: Dynamic memory
			
			attention_mask: Mask for attention
			head_mask: Mask for attention heads
			
			output_attention: Output attention
			output_hidden: Output hidden states
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
		
		if self.drop is not None:
			x = self.drop(x)
		
		output = {
			"hidden": [] if output_hidden else None,
			"attention": [] if output_attention else None
		}
		
		self.transformer(x,
		 	dynamic_memory=dynamic_memory,
			static_memory=static_memory,
			attention_mask=attention_mask,
		 	output=output
		)
		
		if output_hidden:
			output['hidden'].append(x)
		
		# Process final hidden state
		if self.ln_f is not None:
			x = self.ln_f(x)
		
		x = self.lm_head(x)
		
		return x, output

class DMTransformerBlock(nn.Module):
	def __init__(self, embed, eps, attn, smem):
		super().__init__()
		self.ln_1 = nn.LayerNorm(embed, eps)
		self.attn = attn
		self.ln_2 = nn.LayerNorm(embed, eps)
		self.smem = smem
	
	def forward(self, x, **ctx):
		x = self.ln_1(x)
		x = self.attn(x, **ctx)
		x = self.ln_2(x)
		x = self.smem(x, **ctx)
		return x

class DMTransformer(nn.Module):
	'''
	Discrete Memory Transformer model designed to load GPT-2 weights
	'''
	
	def __init__(self,
			embed: int,
			layers: int,
			heads: int,
			max_seq: int,
			
			# Norms
			norm_epsilon: Optional[float]=NORM,
			prenorm_epsilon: Optional[float]=None,
			
			# Dropout
			dropout_p: Optional[float]=DROP,
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
		super().__init__()
		
		prenorm_epsilon = default(prenorm_epsilon, norm_epsilon)
		pdrop_attn = default(pdrop_attn, dropout_p) or None
		
		# Note to self: if we want this to succeed as a proof of concept, we
		#  can't be too experimental with the setup. It needs to be roughly
		#  1:1 with the teacher model, even if that doesn't make complete
		#  sense. Here, MHA/SMA relates directly to GPT-2 MHA/MLP
		def block():
			return DMTransformerBlock(
				embed, prenorm_epsilon,
				MultiheadAttention(embed, heads,
					selector=ScaledDotProductSelector(
						max_seq, dropout=pdrop_attn
					)
				),
				StaticMemoryAttention(embed)
			)
		
		self.h = Residual([
			block() for i in range(1, layers)
		])
	
	def forward(self, x, **ctx):
		for layer in self.h:
			x = layer(x, **ctx)
		return x

class DMGPT2Model(DMTransformer):
	'''
	DMTransformer configured using GPT-2 parameters.
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
			pdrop_attn=config.attn_pdrop,
			
			prenorm_epsilon=config.layer_norm_epsilon,
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