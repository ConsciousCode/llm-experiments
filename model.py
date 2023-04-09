import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from typing import Optional, TypeVar, Callable, Mapping
import knn
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from torch import nn
import torch
import logging

default_config = config.__dict__
logger = logging.getLogger(__name__)

T = TypeVar("T")
def default(x: Optional[T], y: T|Callable[[], T]) -> T:
	'''Extensible defaults for function arguments.'''
	return x if x is not None else y() if callable(y) else y

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

class LearnableTopP(nn.Module):
	'''
	Select the top elements of a tensor based on learnable thresholds.
	'''
	
	def __init__(self, num_heads, threshold, dim=-1):
		super().__init__()
		self.dim = dim
		self.threshold = nn.Parameter(torch.tensor([threshold] * num_heads))
	
	def forward(self, atn):
		batch, num_heads, seq, _ = atn.shape

		# Reshape to merge head and seq dimensions
		atn_combined = atn.view(batch, num_heads * seq, seq)

		# Sort, sum, and mask
		atn, idx = torch.sort(atn_combined, self.dim, descending=True)
		cumsum = torch.cumsum(atn, self.dim)
		mask = (cumsum <= torch.sigmoid(self.threshold).view(1, num_heads, 1).repeat(1, seq, 1))

		# Apply mask and collect indices in a list
		masked_idx = []
		for i in range(batch):
			for j in range(num_heads * seq):
				masked_idx.append(torch.masked_select(idx[i, j], mask[i, j]))
		print(len(masked_idx))
		return masked_idx


class InfoStillAttention(nn.Module):
	'''
	Simple attention with QK-Norm
	'''
	
	def __init__(self,
		  embed: int,
		  max_seq: int,
		  dropout: Optional[nn.Module]=None
		):
		'''
		Parameters:
			embed: Embedding dimension
			max_seq: Maximum positions
			dropout: Dropout module
		'''
		super().__init__()
		
		self.register_buffer("bias",
			torch.tril(torch.ones((max_seq, max_seq), dtype=torch.bool)).view(
				1, 1, max_seq, max_seq
			),
		)
		
		self.embed_dim = embed
		self.qkscale = nn.Parameter(torch.tensor(embed ** -0.5))
		self.dropout = dropout
	
	def forward(self, q, k, v, attn_mask=None, head_mask=None):
		# Query-Key Normalization for Transformers
		# https://arxiv.org/abs/2010.04245
		q = F.normalize(q, p=2, dim=-1)
		k = F.normalize(k, p=2, dim=-1)
		
		attn_weight = torch.matmul(q, k.transpose(-1, -2))
		out_attn = attn_weight
		attn_weight = attn_weight * self.qkscale
		
		dtype, device = attn_weight.dtype, attn_weight.device
		qlen, klen = q.shape[-2], k.shape[-2]
		
		# Causal mask
		causal_mask = self.bias[:, :, klen - qlen : klen, :klen]
		mask_value = torch.finfo(dtype).min
		mask_value = torch.full([], mask_value, dtype=dtype).to(device)
		attn_weight = torch.where(causal_mask, attn_weight, mask_value)

		if attn_mask is not None:
			attn_weight = attn_weight + attn_mask

		attn_weight = F.softmax(attn_weight, dim=-1)

		# Downcast (if necessary) back to V's dtype
		attn_weight = self.dropout(attn_weight.type(v.dtype))

		if head_mask is not None:
			attn_weight = attn_weight * head_mask
		
		return torch.matmul(attn_weight, v), out_attn

class InfoStill(nn.Module):
	'''
	InfoStill: A Memory Augmented Transformer with Feedback
	
	No feedforward implemented because its role is neural memory, which we've
	 replaced with the kNN memory.
	
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
	'''
	
	def __init__(self,
		  embed: int,
		  heads: int,
		  max_seq: int,
		  layer_idx: int,
		  *,
		  pos_embed: bool=False,
		  
		  # Modules
		  attention: Optional[nn.Module]=None,
		  prenorm: Optional[nn.Module]=None,
		  dropout: Optional[nn.Module]=None,
		  
		  # Defaults for modules
		  prenorm_epsilon: Optional[float]=None,
		  dropout_p: Optional[float]=None,
		  pdrop_residual: Optional[float]=None,
		  pdrop_attn: Optional[float]=None
		):
		'''
		Parameters:
			embed: Embedding dimension
			heads: Number of attention heads
			max_seq: Maximum number of positions
			layer_idx: Index of this layer in the stack
			pos_embed: Positional embedding
			
			attention: Attention module (default: InfoStillAttention)
			prenorm: Pre-normalization module (default: LayerNorm using layer_norm_epsilon)
			dropout: Dropout probability for residual connections
			
			layer_norm_epsilon: Layer norm epsilon (default: 1e-5)
			dropout_p: Common dropout probability for all dropout layers
			pdrop_residual: Dropout probability for residual connections
			pdrop_attn: Dropout probability for attention weights
		'''
		super().__init__()

		self.max_seq = max_seq
		self.embed = embed
		self.heads = heads
		self.layer_idx = layer_idx
		
		assert embed % heads == 0
		self.hidden = embed // heads
		
		prenorm_epsilon = default(prenorm_epsilon, 1e-5)
		self.prenorm = default(prenorm, lambda: nn.LayerNorm(embed, prenorm_epsilon))
		self.pos_embed = RotaryEmbedding(embed) if pos_embed else None
		
		# Original GPT-2 uses Conv1D from pytorch_util to take advantage of addmm
		#  which is marginally faster than matmul. This isn't necessary for newer
		#  versions of Torch which automatically use admm for 2D matmul.
		#
		# Cramming: Training a Language Model on a Single GPU in One Day
		# https://arxiv.org/abs/2212.14034
		# * Removing bias has negligible effect on loss and reduces parameters
		self.qkv_proj = nn.Linear(embed, embed * 3, bias=False)
		self.out_proj = nn.Linear(embed, embed, bias=False)
		
		# TODO: Too lazy to make initial values configurable
		self.remember = LearnableTopP(heads, 0.5)
		self.retain = LearnableTopP(heads, 0.5)
		
		dropout_p = default(dropout_p, 0.1)
		pdrop_residual = default(pdrop_residual, dropout_p)
		pdrop_attn = default(pdrop_attn, dropout_p)
		
		self.attention = default(attention, lambda:
			InfoStillAttention(embed, max_seq, nn.Dropout(pdrop_attn))
		)
		self.dropout = default(dropout, lambda: nn.Dropout(pdrop_residual))

	def _split_heads(self, tensor):
		"""
		Splits hidden_size dim into attn_head_size and heads
		"""
		new_shape = tensor.shape[:-1] + (self.heads, self.hidden)
		tensor = tensor.view(new_shape)
		return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

	def _merge_heads(self, tensor):
		"""
		Merges attn_head_size dim and num_attn_heads dim into hidden_size
		"""
		tensor = tensor.permute(0, 2, 1, 3).contiguous()
		new_shape = tensor.shape[:-2] + (self.embed,)
		return tensor.view(new_shape)

	def forward(self,
		x: Optional[tuple[torch.Tensor]],
		*,
		# Feedback
		memory: Optional[object]=None,
		layer_past: Optional[tuple[torch.Tensor]]=None,
		
		# Masks
		attention_mask: Optional[torch.Tensor]=None,
		head_mask: Optional[torch.Tensor]=None,
		
		# Flags
		use_cache=False,
		output_attentions=False
	) -> tuple[torch.Tensor | tuple[torch.Tensor], ...]:
		'''
		Parameters:
			x: Hidden states from the previous layer
			layer_past: (key, value) from previous time steps
			
			attention_mask: Mask to avoid performing attention on padding token indices
			head_mask: Mask to nullify selected heads of the self-attention modules
			
			use_cache: Whether or not to use the cached key/value states
			output_attentions: Whether or not to return the attentions tensors of all attention layers
		
		Return: output, present, (attentions)
		'''
		
		batch, seq, embed = x.shape
		
		x = self.prenorm(x)
		
		q, k, v = self.qkv_proj(x).split(self.embed, dim=2)
		
		print("before qkv", q.shape, k.shape, v.shape)
		if self.pos_embed is not None:
			q, k = self.pos_embed(q, k)
		
		print("after qkv", q.shape, k.shape, v.shape)
		q, k, v = map(self._split_heads, (q, k, v))
		
		# Caching for faster inference
		if layer_past is not None:
			past_k, past_v = layer_past
			k = torch.cat((past_k, k), dim=-2)
			v = torch.cat((past_v, v), dim=-2)

		present = (k, v) if use_cache else None
		
		latn, latn_weight = self.attention(q, k, v, attention_mask, head_mask)
		#print("latn", latn_weight.shape, latn_weight)
		# Use similarity to select memory queries/keys
		#latn_weight = latn_weight.view(batch, seq, embed)
		print(latn_weight.shape)
		print("qkv", q.shape, k.shape, v.shape)
		
		# Find the index with the maximum value for each head along the last dimension
		max_values, max_idx = torch.max(latn_weight, dim=-1)

		# Find the index with the maximum value for each head along the sequence dimension
		max_values_seq, max_idx_seq = torch.max(max_values, dim=-1)

		# Get the indices of the maximum attention weights per head
		max_indices = max_idx.gather(dim=-1, index=max_idx_seq.unsqueeze(-1)).squeeze(-1)

		# Prepare the indices for gather operation
		max_indices = max_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, q.shape[-1])

		print("shape", max_values.shape, max_values_seq.shape, max_indices.shape)
		print("1", max_values)
		print("2", max_values_seq)
		print("3", max_indices)

		# Gather the corresponding queries, keys, and values based on the maximum attention weight indices
		mq = q.gather(dim=2, index=max_indices)
		mk = k.gather(dim=2, index=max_indices)
		mv = v.gather(dim=2, index=max_indices)
		
		print(mq, mk, mv)
		print("m qkv", mq, mk, mv)

		# Tomorrow TODO: mq is getting (1, heads, 1, 0)
		#mq, mk, mv = map(self._merge_heads, (mq, mk, mv))
		
		# Query-and-update memory operation
		mk, mv = knn.ExternalMemory.apply(mq, mk, mv, memory)

		# Compute memory attention, no masking
		matn, matn_weights = self.attention(q, mk, mv)

		# Combine attentions
		attn_out = latn + matn
		attn_weight = torch.cat((latn_weight, matn_weights), dim=-1)
		
		attn_out = self._merge_heads(attn_out)
		attn_out = self.out_proj(attn_out)
		
		if self.dropout is not None:
			attn_out = self.dropout(attn_out)

		outputs = (attn_out, present)
		if output_attentions:
			outputs += (attn_weight,)

		return outputs

class InfoDistillery(nn.Module):
	'''
	Container for InfoStill layers. Keeps track of feedback and memory.
	
	Addressing Some Limitations of Transformers with Feedback Memory
	https://arxiv.org/abs/2002.09402
	* Using output of upper layers for lower (modified: per layer pair, no pooling)
	
	Memory transformers
	https://arxiv.org/abs/2006.11527
	* Concatenating memory to input tokens (modified: no memory controller)
	'''
	
	def __init__(self,
		  token_embed: nn.Module,
		  layers: list[nn.Module],
		  dropout: Optional[nn.Module]=None,
		  postnorm: Optional[nn.Module]=None,
		  dtype: Optional[torch.dtype]=None
		):
		'''
		Parameters
			layers - The constructed layers
		'''
		super().__init__()
		
		self.token_embed = token_embed
		self.layers = nn.ModuleList(filter(bool, layers))
		self.dropout = dropout
		self.postnorm = postnorm
		self.dtype = default(dtype, torch.float32)
		
	def get_input_embeddings(self):
		return self.token_embed
	
	def set_input_embeddings(self, value):
		self.token_embed = value
	
	def forward(self,
		  x: torch.Tensor,
		  *,
		  memory: Optional[object]=None,
		  feedback: Optional[list[torch.Tensor]]=None,
		  past_kv: Optional[tuple[torch.Tensor, torch.Tensor]]=None,
		  
		  attention_mask: Optional[torch.Tensor]=None,
		  head_mask: Optional[torch.Tensor]=None,
		  
		  input_embeds=False,
		  output_attentions=False,
		  output_hidden_states=False,
		  use_cache=False
		) -> tuple:
		'''
		kwargs:
			x: Input, either ids or embeddings
			feedback: Feedback from previous timestep
			past_kv: key and value for past attention
			
			attention_mask: Mask for attention
			head_mask: Mask for attention heads
			
			imput_embeds: Input embeddings instead of ids
			output_attentions: Output attentions
			output_hidden_states: Output hidden states
			use_cache: Use cache for attention
		'''
		
		print("x shape", x.shape)
		
		if not input_embeds:
			x = self.token_embed(x)
		
		past_kv = default(past_kv, lambda: (None,) * len(self.layers))
		
		batch = x.shape[0]

		# GPT2Attention mask
		if attention_mask is not None:
			attention_mask = attention_mask.view(batch, -1)
			attention_mask = attention_mask[:, None, None, :]
			
			# Adjust to (-inf, 0] for additive mask
			attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
			attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min
		
		if use_cache and self.gradient_checkpointing and self.training:
			logger.warning_once(
				"`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
			)
			use_cache = False
		
		output_shape = x.shape
		
		if self.dropout is not None:
			x = self.dropout(x)
		
		presents, all_attn, all_hidden = [], [], []
		
		# Layers
		for i, (layer, past) in enumerate(zip(self.layers, past_kv)):
			if output_hidden_states:
				all_hidden.append(x)
			
			# Apply feedback
			if feedback is not None and feedback[i] is not None:
				#x = torch.cat((feedback[i], x), dim=1)
				x = x + feedback[i]
			
			hmask = head_mask and head_mask[i]
			
			# Apply the layer
			outputs = layer(x,
		   		memory=memory,
				layer_past=past,
				attention_mask=attention_mask,
				head_mask=hmask,
				use_cache=use_cache,
				output_attentions=output_attentions,
			)
			x, attns, *cache = outputs
			
			if use_cache:
				presents.append(attns)
				attns = cache[0]
			
			if output_attentions:
				all_attn.append(attns)
		
		# Process final hidden state
		if self.postnorm is not None:
			x = self.postnorm(x).view(output_shape)
		
		if output_hidden_states:
			all_hidden.append(x)
		
		out = (presents, all_hidden, all_attn)
		return (x, *filter(len, out))

class InfoDistilleryModel(InfoDistillery):
	'''
	Specific instantiation of the InfoDistillery architecture, manages
	creating the layers as needed using configuration.
	'''
	
	def __init__(self,
		  vocab: int,
		  embed: int,
		  layers: int,
		  heads: int,
		  max_seq: int,
		  prenorm_epsilon: Optional[float]=None,
		  postnorm_epsilon: Optional[float]=None,
		  dropout_p: Optional[float]=1e-1,
		  pdrop_embed: Optional[float]=None,
		  pdrop_residual: Optional[float]=None,
		  pdrop_attn: Optional[float]=None
		):
		'''
			Parameters:
				vocab - Vocabulary size
				embed - Embedding size
				layers - Number of layers
				heads - Number of heads
				max_seq - Maximum sequence length
				dropout_p - Dropout rate
				pdrop_embed - Embedding dropout rate
				pdrop_residual - Residual dropout rate
				pdrop_attn - Attention dropout rate
				prenorm_epsilon - Layer norm epsilon
		'''
		
		config = {
			"embed": embed,
			"heads": heads,
			"max_seq": max_seq,
			"prenorm_epsilon": prenorm_epsilon,
			"dropout_p": dropout_p,
			"pdrop_residual": pdrop_residual,
			"pdrop_attn": pdrop_attn
		}
		
		pdrop_embed = default(pdrop_embed, dropout_p)
		
		super().__init__(
			nn.Embedding(vocab, embed), [
				InfoStill(**config, layer_idx=0, pos_embed=True),
				*(InfoStill(**config, layer_idx=i)
					for i in range(1, layers))
			],
			dropout=dropout_p and nn.Dropout(pdrop_embed),
			postnorm=nn.LayerNorm(embed, default(prenorm_epsilon, 1e-5))
		)
		config.update({
			"vocab": vocab,
			"layers": layers,
			"postnorm_epsilon": postnorm_epsilon,
			"pdrop_embed": pdrop_embed
		})
		self.config = config

class InfoDistilleryGPT2Model(InfoDistilleryModel):
	'''
	InfoDistilleryModel configured using GPT-2 parameters.
	'''
	
	def __init__(self, config: GPT2Config):
		'''
		Relevant config parameters:
			vocab_size (`int`, *optional*, defaults to 50257):
				Vocabulary size of the GPT-2 model. Defines the number of different tokens that can be represented by the
				`inputs_ids` passed when calling [`GPT2Model`] or [`TFGPT2Model`].
			n_positions (`int`, *optional*, defaults to 1024):
				The maximum sequence length that this model might ever be used with. Typically set this to something large
				just in case (e.g., 512 or 1024 or 2048).
			n_embd (`int`, *optional*, defaults to 768):
				Dimensionality of the embeddings and hidden states.
			n_layer (`int`, *optional*, defaults to 12):
				Number of hidden layers in the Transformer encoder.
			n_head (`int`, *optional*, defaults to 12):
				Number of attention heads for each attention layer in the Transformer encoder.
			resid_pdrop (`float`, *optional*, defaults to 0.1):
				The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
			embd_pdrop (`float`, *optional*, defaults to 0.1):
				The dropout ratio for the embeddings.
			attn_pdrop (`float`, *optional*, defaults to 0.1):
				The dropout ratio for the attention.
			layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
				The epsilon to use in the layer normalization layers.
		'''
		super().__init__(
			vocab=config.vocab_size,
			embed=config.n_embd,
			layers=config.n_layer,
			heads=config.n_head,
			max_seq=config.n_positions,
			dropout_p=config.resid_pdrop,
			pdrop_embed=config.embd_pdrop,
			pdrop_residual=config.resid_pdrop,
			pdrop_attn=config.attn_pdrop,
			prenorm_epsilon=config.layer_norm_epsilon,
			postnorm_epsilon=config.layer_norm_epsilon
		)

class InfoDistilleryLMHeadModel(nn.Module):
	'''
	Wraps a given model with a linear layer to convert the hidden state to
	logits, tied to the embedding matrix.
	'''
	
	def __init__(self, model):
		super().__init__()
		
		self.model = model
		embed = model.get_input_embeddings()
		
		self.lm_head = nn.Linear(embed.num_embeddings, embed.embedding_dim, bias=False)
		self.lm_head.weight = embed.weight
	
	def forward(self, x, **config):
		# Convert hidden state to logits
		output = self.model(x, **config)
		hidden_states = output[0]
		logits = self.lm_head(hidden_states)
		return (logits,) + output[1:]

class TransformersWrapper(nn.Module):
	'''
	Wrapper converting transformers-style inputs to InfoDistillery-style inputs.
	'''
	
	def __init__(self, model):
		super().__init__()
		
		self.model = model
	
	def forward(self, *, input_ids, **kwargs):
		return self.model(input_ids, **kwargs)