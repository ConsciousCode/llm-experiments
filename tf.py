#!/usr/bin/env python3
'''
Common transformer classes and functions.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, TypeAlias, Type
from abc import abstractmethod
from dataclasses import dataclass, field

from common import NORM, DROP, BIAS, default

ConfigMod: TypeAlias = Optional[bool|int|float|nn.Module]

def mod_or_config(config: ConfigMod, default: float, mod: Type[nn.Module]):
	'''Returns a module or a module initialized with a config.'''
	
	if isinstance(config, (int, float)):
		return mod(config) if config else None
	if isinstance(config, bool):
		return mod(default) if config else None
	return config

@dataclass
class Output:
	'''Optional outputs for layers.'''
	attention: Optional[list[Optional[torch.Tensor]]] = None
	'''Attention weights of each layer'''
	cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None
	'''Key-value cache for faster inference'''

@dataclass
class Context:
	'''Context for layers.'''
	output: Output = field(default_factory=Output)
	'''Optional outputs for layers.'''
	feedback: Optional[list[torch.Tensor]] = None
	'''Feedback for instill.'''
	mask: Optional[torch.Tensor] = None
	'''Attention mask.'''
	head_mask: Optional[torch.Tensor] = None
	'''Attention head mask.'''
	cache: Optional[torch.Tensor] = None
	'''Key-value cache for faster inference'''

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

	def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
	'''Traditional softmax(Q K^T) V attention selector.'''
	
	def __init__(self, max_seq: Optional[int], dropout: ConfigMod=None):
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
	
	def forward(self,
	    	q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
			ctx: Context
		) -> torch.Tensor:
		attn_weight = torch.matmul(q, k.transpose(-1, -2))
		
		dtype, device = attn_weight.dtype, attn_weight.device
		qlen, klen = q.shape[-2], k.shape[-2]
		
		# Causal mask
		if self.bias is not None:
			causal_mask = self.bias[:, :, klen - qlen : klen, :klen]
			mask_value = torch.finfo(dtype).min
			mask_value = torch.full((), mask_value, dtype=dtype).to(device)
			attn_weight = torch.where(causal_mask, attn_weight, mask_value)
			
			attn_mask = ctx.mask
			if attn_mask is not None:
				attn_weight = attn_weight + attn_mask
		
		attn_weight = F.softmax(attn_weight, dim=-1)
		
		# Downcast (if necessary) back to V's dtype
		if self.attn_dropout is not None:
			attn_weight = self.attn_dropout(attn_weight.type(v.dtype))
		
		head_mask = ctx.head_maskcausal_mask
		if head_mask is not None:
			attn_weight = attn_weight * head_mask
		
		attn = torch.matmul(attn_weight, v)
		
		atv = ctx.output.attention
		if atv is not None:
			atv.append(attn_weight)
		
		return attn

class AssociativeMemorySelector(nn.Module):
	'''
	Attention recontextualized as kNN memory. Replaces feed forwarde layers.
	1. QKV projections
	2. QK L2 norm, KV STE
	3. Top-K for keys and their values
	4. softmax(Q K^T) V attention on the results
	
	Transformer Feed-Forward Layers Are Key-Value Memories
	https://arxiv.org/abs/2012.14913
	
	Augmenting Self-attention with Persistent Memory
	https://arxiv.org/pdf/1907.01470.pdf
	* Proves FF networks are equivalent to attention with static memory
	
	Attention Approximates Sparse Distributed Memory
	https://arxiv.org/abs/2111.05498
	* Theoretical basis for why FF might be both attention and memory
	
	Memorizing Transformers
	https://arxiv.org/abs/2203.08913
	* kNN memory, paper uses it as an alternative to recurrency
	
	Neural Turing Machines
	https://arxiv.org/abs/1410.5401
	* Read and write/erase heads, paper uses it for memory-augmented tasks
	'''
	
	def __init__(self,
	    	embed,
			selector: Optional[nn.Module]=None,
			max_seq: Optional[int]=None,
			dropout: ConfigMod=None
		):
		'''
		Parameters:
			embed: Embedding dimension
			selector: Selector module after memory lookup
			max_seq: Maximum sequence length
			dropout: Residual dropout
		'''
		
		super().__init__()
		self.selector = default(selector, lambda: ScaledDotProductSelector(max_seq, dropout))
		
		# sqrt of QK Norm init because we can't assume selector is a dot product
		self.scale = nn.Parameter(torch.tensor(embed ** -0.25))
	
	def forward(self,
	    	q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
			ctx: Context
		) -> torch.Tensor:
		# QK Norm and STE
		q = F.normalize(q, dim=-1)
		k = F.normalize(k, dim=-1).detach()
		v = v.detach()
		
		mk, mv = ctx.associative.search(k, v)
		
		# STE
		k.grad = mk.grad
		v.grad = mv.grad
		
		q, k = q * self.scale, k * self.scale
		return self.selector(q, k, v, ctx)

class Residual(nn.ModuleList):
	'''
	Adds residuals to each sequential model.
	'''
	
	def forward(self, x, ctx):
		for layer in self:
			x = x + layer(x, ctx)
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
	
	def _feedback(self, x, f, ctx):
		'''How to apply the feedback'''
		return x if f is None else x + f
	
	def forward(self, x, ctx):
		f = ctx.feedback
		if f is None:
			f = [None] * len(self)
		
		for i, layer in enumerate(self):
			print("Layer", i)
			x = x + layer(self._feedback(x, f[i], ctx), ctx)
		return x

class TransformerLayer(nn.Module):
	'''
	Base class for layers in a transformer. Mostly norms and dropout,
	since some forms of attention don't even have the QKV projections.
	
	Attention Is All You Need
	https://arxiv.org/abs/1706.03762
	'''
	
	def __init__(self,
			embed: int,
			*,
			prenorm: ConfigMod=None,
			dropout: ConfigMod=None,
			postnorm: ConfigMod=None,
			bias: bool=BIAS
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
		
		self.prenorm = mod_or_config(prenorm, NORM, lambda eps: nn.LayerNorm(embed, eps))
		self.resid_dropout = mod_or_config(dropout, DROP, nn.Dropout)
		self.postnorm = postnorm and mod_or_config(postnorm, NORM, lambda eps: nn.LayerNorm(embed, eps))
		self.out_proj = nn.Linear(embed, embed, bias)
	
	@abstractmethod
	def _forward(self, x: torch.Tensor, ctx: Context) -> torch.Tensor:
		'''
		Perform attention-specific operations to convert input to Q, K, and V,
		then into an attention output.
		'''
		pass
	
	def forward(self, x: torch.Tensor, ctx: Context) -> torch.Tensor:
		'''
		Normalizations, projections, and dropouts.
		'''
		
		if self.prenorm is not None:
			x = self.prenorm(x)
				
		atv = ctx.output.attention
		atn = self._forward(x, ctx)
		if atv is not None:
			assert len(ctx.output.attention) > len(atv), "Attention output is not growing"
		
		atn = self.out_proj(atn)
		
		if self.postnorm is not None:
			atn = self.postnorm(atn)
		
		if self.resid_dropout is not None:
			atn = self.resid_dropout(atn)
		
		if h := ctx.output.hidden:
			h.append(atn)
		
		return atn

class MultiheadAttention(TransformerLayer):
	'''
	Normal mutli-headed attention with qkv and output projections.
	'''
	
	def __init__(self,
			embed: int,
			heads: int=1,
			*,
			rotary_embed: bool=False,
			max_seq: Optional[int]=None,
			selector: Optional[nn.Module]=None,
			
			dropout: ConfigMod=None,
			bias: bool=BIAS
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
			dropout=dropout,
			bias=bias
		)
		self.heads = heads
		
		self.rotary = RotaryEmbedding(embed) if rotary_embed else None
		
		if selector is None:
			assert max_seq is not None, "max_seq must be specified if selector is not"
			selector = ScaledDotProductSelector(max_seq)
		self.selector = selector
		
		self.scale = embed ** -0.25
		
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
	
	def _forward(self, x, ctx):
		q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
		
		if self.rotary is not None:
			q, k = self.rotary(q, k, ctx)
		
		# Caching for faster inference
		cache = ctx.cache
		if cache is not None:
			past_k, past_v = cache
			k = torch.cat((past_k, k), dim=-2)
			v = torch.cat((past_v, v), dim=-2)
			ctx.output.cache = (k, v)
		
		# QK Norm
		q = F.normalize(q, dim=-1) * self.scale
		k = F.normalize(k, dim=-1) * self.scale
		
		q, k, v = map(self._split_heads, (q, k, v))
		x = self.selector(q, k, v, ctx)
		return self._merge_heads(x)

class LanguageModel(nn.Module):
	'''
	Wraps a language model with a token embedding and a linear output layer.
	'''
	
	def __init__(self,
			vocab: int,
			embed: int,
			model: list[nn.Module],
			dropout: ConfigMod=None,
			postnorm: ConfigMod=None,
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
		self.embed_dropout = mod_or_config(dropout, DROP, nn.Dropout)
		self.lm = model
		self.postnorm = mod_or_config(postnorm, NORM, lambda x: nn.LayerNorm((embed,), x))
		# Tie lm_head and embed weights
		self.lm_head = nn.Linear(vocab, embed, bias=False)
		self.dtype = dtype or torch.float32
	
	def tie_weights(self):
		'''Tie lm_head and embed weights'''
		self.lm_head.weight = self.embed.weight
	
	def forward(self, x: torch.Tensor, ctx: Context) -> torch.Tensor:
		'''
		kwargs:
			x: Input embeddings or ids
			featural: Static memory
			associative: Dynamic memory
			
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
		
		if self.embed_dropout is not None:
			x = self.embed_dropout(x)
		
		x = self.lm(x, ctx)
		
		hidden = ctx.output.hidden
		if hidden is not None:
			hidden.append(x)
		
		# Process final hidden state
		if self.postnorm is not None:
			x = self.postnorm(x)
		
		x = self.lm_head(x)
		return x