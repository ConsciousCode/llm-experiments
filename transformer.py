'''
Defining high level types which make working with transformers a lot easier.

Attention Is All You Need
https://arxiv.org/abs/1706.03762
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from array import array

from dataclasses import dataclass, astuple
from typing import Optional, Mapping, overload, TypeVar, Callable, Literal, Sequence, TypeAlias
from abc import abstractmethod

import knn
import rotary_embedding_torch

#######################
## Utility functions ##
#######################

T = TypeVar("T")
def default(x: Optional[T], y: T|Callable[[], T]) -> T:
	'''Extensible defaults for function arguments.'''
	return x if x is not None else y() if callable(y) else y

def skip_none(*args):
	yield from (x for x in args if x is not None)

#####################
## Utility classes ##
#####################

@dataclass
class KVEmbed:
	'''
	An object containing keys and values, makes for easier reasoning since
	they're almost always together until the final attention operation.
	'''
	
	key: torch.Tensor
	value: torch.Tensor
	
	@overload
	def __init__(self, kv: 'KVEmbed'): ...
	@overload
	def __init__(self, k: torch.Tensor, v: torch.Tensor): ...
	
	def __init__(self, k, v = None):
		if v is None:
			k, v = k
		super().__init__(k, v)
	
	def __iter__(self):
		return iter(astuple(self))

@dataclass
class QKVEmbed:
	'''
	An object containing query and key-values, also for easier reasoning.
	'''
	
	query: torch.Tensor
	kv: KVEmbed
	
	@overload
	def __init__(self, qkv: 'QKVEmbed'): ...
	@overload
	def __init__(self, q: torch.Tensor, kv: KVEmbed): ...
	@overload
	def __init__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor): ...
	
	def __init__(self, q, k=None, v=None):
		if k is None:
			q, k, v = q
		super().__init__(q, KVEmbed(k, v))
	
	def __iter__(self):
		return iter((self.query, *self.kv))
	
	@property
	def key(self):
		return self.kv.keys
	@key.setter
	def key(self, value):
		self.kv.keys = value
	
	@property
	def value(self):
		return self.kv.value
	@value.setter
	def value(self, value):
		self.kv.value = value

###########################
## Abstract base classes ##
###########################

class Projection(nn.Module):
	'''
	Abstraction of the projection mechanism and any required parameters.
	'''
	
	@abstractmethod
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		'''
		Perform the projection.
		'''
		pass

class Attention(nn.Module):
	'''
	Abstraction of the attention mechanism and any required parameters.
	'''
	
	@abstractmethod
	def forward(self, qkv: QKVEmbed) -> torch.Tensor:
		'''
		Taking a QKV embedding (projected from the input and modified as needed),
		recombine them into the attention.
		'''
		pass

class Span(nn.Module):
	'''
	Some kind of memory trimming. Intended for learnable trimming.
	'''
	
	@abstractmethod
	def trim(self, x: torch.Tensor) -> torch.Tensor:
		'''
		Trim memory as needed.
		'''
		pass
	
	@abstractmethod
	def mask(self, x: torch.Tensor) -> torch.Tensor:
		'''
		Apply masking.
		'''
		pass
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.mask(x)

class PositionalEmbed(nn.Module):
	'''
	An embedding which somehow changes the QKVEmbed to encode position.
	'''
	
	@abstractmethod
	def forward(self, qkv: QKVEmbed) -> QKVEmbed:
		'''
		Implement the positional embedding.
		'''
		pass

#####################
## Implementations ##
#####################

## Losses ##

def hinge_loss(y_pred: torch.Tensor, y_true: torch.IntTensor, margin=1):
	"""
	Multi-class hinge loss.
	
	Parameters
		y_pred: Predicted logits, shape (batch_size, num_classes)
		y_true: True labels, shape (batch_size,)
		margin: The margin for the hinge loss
	"""
	
	batch_size = y_pred.shape[0]
	# Logits for the correct class for each sample in the batch 
	correct_logits = y_pred[torch.arange(batch_size), y_true].unsqueeze(1)
	margins = margin - correct_logits + y_pred
	# max(0, margin - y_pred*(y_true - 1))
	losses = torch.clamp(margins, min=0)
	# loss=0 for correct classes
	losses[torch.arange(batch_size), y_true] = 0
	# Average sum of the losses across the batch
	return torch.mean(torch.sum(losses, dim=1))

def confidence_penalty(y_pred):
	"""
	Penalizes overconfident predictions based on the entropy of the logits,
	leads to better generalization and reduced overfitting.
	
	Regularizing Neural Networks by Penalizing Confident Output Distributions
	https://arxiv.org/abs/1701.06548
	
	Parameters
		y_pred: Predicted logits, shape (batch_size, num_classes)
	"""
	P = F.softmax(y_pred, dim=1) * F.log_softmax(y_pred, dim=1)
	entropy = -torch.sum(P, dim=1)
	return torch.mean(entropy)

## Norms ##

## Meta-layers

class PreNorm(nn.Module):
	'''
	Meta-layer which normalizes the input before the network.
	'''
	
	def __init__(self, net: nn.Module, norm: nn.Module):
		super().__init__()
		
		self.norm = norm
		self.net = net
	
	def forward(self, x):
		return self.net(self.norm(x))

class PostNorm(nn.Module):
	'''
	Meta-layer which normalizes the output after the network.
	'''
	
	def __init__(self, net: nn.Module, norm: nn.Module):
		super().__init__()
		self.net = net
		self.norm = norm
	
	def forward(self, x):
		return self.norm(self.net(x))

class Residual(nn.Module):
	'''
	Meta-layer which adds the residual to the network output.
	'''
	
	def __init__(self, net):
		super().__init__()
		self.net = net
	
	def forward(self, x):
		return self.net(x) + x

## Norm layers

class FixNorm(nn.Module):
	'''
	FixNorm(x) = x/|x|
	
	Used in a lot of papers, but name is from
	
	Improving Lexical Choice in Neural Machine Translation
	https://arxiv.org/abs/1710.01329	'''
	
	def __init__(self, eps: Optional[float]=None):
		super().__init__()
		self.eps = default(eps, 1e-5)
	
	def forward(self, x):
		return F.normalize(x, dim=-1).clamp(min=self.eps)

class ScaleNorm(FixNorm):
	'''
	Fix normalization with a learnable scale parameter.
	
	ScaleNorm(x; g) = g*x/|x|
	
	(from Transformers Without Tears - see TearlessNorm)
	'''
	
	def __init__(self, scale: Optional[float]=None, eps: Optional[float]=None):
		'''
		Parameters
			scale - initial scale, the code of the paper suggests sqrt(emb_dim)
		'''
		
		super().__init__(eps)
		self.scale = nn.Parameter(torch.tensor(default(scale, 1)))
	
	def forward(self, x):
		return self.scale * super().forward(x)

class TearlessNorm(nn.Module):
	'''
	Combined PreNorm, FixNorm, and ScaleNorm
	
	Transformers Without Tears: Improving the Normalization of Self-Attention
	https://arxiv.org/abs/1910.05895
	'''
	
	def __init__(self, dims, net, scale: Optional[float]=None):
		super().__init__()
		
		# Could probably combine Residual, PostNorm, PreNorm, LayerNorm,
		#  ScaleNorm, and FixNorm as a single net, but that's less readable
		
		self.net = net
		self.prenorm = nn.LayerNorm((dims,))
		self.fixnorm = FixNorm()
		self.scalenorm = ScaleNorm(scale)
	
	def forward(self, x):
		y = self.net(self.prenorm(x))
		y = self.scalenorm(y)
		return self.fixnorm(x + y)

class QKNorm(nn.Module):
	'''
	Query-Key Normalization for Transformers
	https://arxiv.org/abs/2010.04245
	'''
	
	def __init__(self, scale: Optional[float]=None, dim: Optional[int]=None):
		super().__init__()
		if scale is None:
			scale = default(scale, 1 if dim is None else dim ** -0.5)
		
		self.scale = nn.Parameter(torch.tensor(scale))
	
	def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		scale = torch.sqrt(self.scale)
		q = F.normalize(q, dim=-1) * scale
		k = F.normalize(k, dim=-1) * scale
		return q, k

## Feed-forward layers ##

class FeedForward(nn.Module):
	'''
	Ordinary 2-layer feed forward network from traditional transformer
	
	Removing bias has negligible effect on loss
	
	Cramming: Training a Language Model on a Single GPU in One Day
	https://arxiv.org/abs/2212.14034
	'''
	def __init__(self,
		  dims: int,
		  hidden: Optional[int]=None,
		  activation: Optional[nn.Module]=None,
		  dropout: Optional[nn.Module]=None,
		  bias=False
		):
		super().__init__()
		
		# Hidden layer is recommended to be 4x dimensionality
		hidden = default(hidden, lambda: dims*4)
		
		self.W = nn.ModuleList([
			nn.Linear(dims, hidden, bias),
			nn.Linear(hidden, dims, bias)
		])
		self.activation = activation
		self.dropout = dropout
	
	def forward(self, x):
		x = self.W[0](x)
		x = self.activation(x)
		
		if self.dropout:
			x = self.dropout(x)
		
		return self.W[1](x)

class GLU(nn.Module):
	'''
	Gated Linear Unit, advising Swish as the activation.
	
	SwiGLU, performs better than other activations with the same parameter count.
	
	SwiGLU(x, W, V, b, c, β) = swish(xW + b) ⊗ (xV + c)
	swish(x) = x sigmoid(βx), β is usually 1
	
	Using Torch Hardswish approximation
	
	GLU Variants Improve Transformer
	https://arxiv.org/pdf/2002.05202.pdf
	'''
	def __init__(self,
		  dims: int,
		  hidden: Optional[int]=None,
		  activation: Optional[nn.Module]=None,
		  dropout: Optional[nn.Module]=None,
		  bias=False
		):
		super().__init__()
		
		# Hidden layer is recommended to be 4x dimensionality
		hidden = default(hidden, lambda: dims*4)
		
		# Hidden is doubled to combine the weights of the GLU
		self.W = nn.ModuleList([
			nn.Linear(dims, hidden*2, bias),
			nn.Linear(hidden*2, dims, bias)
		])
		self.activation = activation
		self.dropout = dropout
	
	def forward(self, x):
		x = self.W[0](x)
		x, gate = x.chunk(2, dim = -1)
		x = x * self.activation(gate)
		
		if self.dropout:
			x = self.dropout(x)
		
		return self.W[1](x)

## Attention ##

class Projection(nn.Module):
	'''
	Transformer projection layer, contains the linear layers for calculating
	the projections (eg query, key, and value) necessary for the attention
	mechanism. This is normally part of an attention layer, but many
	transformer augmentations modify the projections before they're combined
	by the actual attention mechanism, which itself has variations. To
	accomodate this, they're treated as two distinct layers. This is an
	abstract projection, so it supports an arbitrary number of projections.
	I use this in InfoStill for a gating projection which controls the
	keys and values which participate in add/update/erase operaitons in an
	external memory.
	'''
	
	def __init__(self,
		  embed: int,
		  hidden: int,
		  heads: int,
		  projections=3,
		  mask: Optional[nn.Module]=None,
		  dropout: Optional[nn.Module]=None,
		  bias=False
		):
		'''
		Parameters:
			embed - The embedding size, usually tied to `hidden`
			hidden - The hidden dimension of the heads per head
			heads - The number of heads, defaults to 1
			projections - Number of projections per head (Q, K, V, etc)
			mask - Optional mask
			dropout - Optional dropout
			saturation - The saturation point of the sigmoid, defaults to 0.9
			growing - Whether or not to allow growing, defaults to True
			bias - Whether or not to include biases, defaults to False
		'''
		super().__init__()
	
		self.embed = embed
		self.hidden = hidden
		self.heads = heads
		self.projections = projections
		self.mask = mask
		self.dropout = dropout
		self.bias = bias

		self.project = nn.Linear(embed, hidden * heads * projections, self.bias)

	def forward(self, x):
		# Project the input to Q, K, and V using the QKV projection layer
		qkv = self.project(x)

		# Split the heads
		qkv = qkv.view(*qkv.size()[:-1], self.heads, self.projections, -1)
		qkv = qkv.permute(0, 2, 1, 3, 4)

		return QKVEmbed(qkv.chunk(3, dim=-2))

class HydraProjection(nn.Module):
	'''
	Growable multi-headed projection. Adds a per-head weight parameter gated
	by sigmoid which, when saturated, indicates more heads are needed. This
	will increase the hidden dimension as needed to evenly split among the
	heads.
	'''
	
	def __init__(self,
		  embed: int,
		  hidden: int,
		  heads: int,
		  projections=3,
		  mask: Optional[nn.Module]=None,
		  dropout: Optional[nn.Module]=None,
		  saturation=0.9,
		  growing=True,
		  bias=False
		):
		'''
		Parameters:
			embed - The embedding size, usually tied to `hidden`
			hidden - The hidden dimension of the heads per head
			heads - The number of heads, defaults to 1
			projection - Number of projections per head (Q, K, V, etc)
			mask - Optional mask
			dropout - Optional dropout
			saturation - The saturation point of the sigmoid, defaults to 0.9
			growing - Whether or not to allow growing, defaults to True
			bias - Whether or not to include biases, defaults to False
		'''
		super().__init__(embed, hidden, heads, projections, mask, dropout, bias)
		
		self.saturation = default(saturation, 0.9)
		self.growing = growing
	
	def grow(self):
		'''
		Check if heads are overburdened and grow if necessary. Note that the
		hidden dimension will not be changed. Enables a model to learn the optimal
		number of heads per layer. To do this, it adds a contribution parameter
		which weights the contributions of the heads. When there aren't enough
		heads, these will tend towards saturating sigmoid at 1 as they
		overcompensate.
		
		Bases for attention head growing:
		
		Are Sixteen Heads Really Better than One?
		https://arxiv.org/abs/1905.10650
		https://blog.ml.cmu.edu/2020/03/20/are-sixteen-heads-really-better-than-one/
		* Ablation shows heads are highly redundant and more important in some layers
		
		Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned
		https://arxiv.org/abs/1905.09418
		'''
		
		if self.growing and self.contrib.sigmoid().mean() > self.saturation:
			self.heads += 1
			
			embed, hidden, heads, proj = self.embed, self.hidden, self.heads, self.projections
			scale = (hidden * heads) ** -0.5
			
			old_QKV = self.QKV
			old_output = self.output

			# Grow the weights
			new_QKV = torch.randn((embed, hidden * proj)) * scale
			new_output = torch.randn((hidden * proj, embed)) * scale

			# Grow the linear modules
			self.QKV = nn.Linear(embed, hidden * heads * proj, self.bias)
			self.output = nn.Linear(hidden * heads * proj, embed, self.bias)

			# Update the weights
			self.QKV.weight = nn.Parameter(torch.cat((old_QKV.weight, new_QKV), dim=0))
			self.output.weight = nn.Parameter(torch.cat((old_output.weight, new_output), dim=0))

			# Grow the bias if it exists
			if self.bias:
				new_bias = torch.zeros(self.hidden * proj)
				self.QKV.bias = nn.Parameter(torch.cat((old_QKV.bias, new_bias), dim=0))
				self.output.bias = nn.Parameter(torch.cat((old_output.bias, torch.zeros(self.embed)), dim=0))

			# Grow the contribution
			self.contrib = nn.Parameter(torch.cat((self.contrib, torch.zeros(1))))

	def grow_hidden(self, new_hidden):
		'''
		Grow the head dimensionality - there's no way to determine this
		automatically, but this lets you grow later.
		
		Not really examined in-depth, might not work as-is
		'''
		assert new_hidden > self.hidden
		
		embed, hidden, heads, proj = self.embed, self.hidden, self.heads, self.projections
		
		diff = new_hidden - hidden
		scale = (hidden * heads) ** -0.5
		self.hidden = new_hidden

		# Modify the QKV layer
		old_QKV = self.QKV
		
		new_QKV = torch.randn(embed, heads * proj * diff) * scale
		new_QKV = new_QKV.view(embed, heads, proj, diff)

		cat_QKV = torch.cat((old_QKV.weight.view(embed, heads, proj, -1), new_QKV), dim=-1)
		cat_QKV = cat_QKV.view(embed, heads * proj* new_hidden)
		self.QKV = nn.Linear(embed, heads * hidden * proj, self.bias)
		self.QKV.weight = nn.Parameter(cat_QKV)

		if self.bias:
			self.QKV.bias = nn.Parameter(old_QKV.bias.clone())

		# Modify the output layer
		new_output = torch.randn(heads * diff, embed) * scale
		new_output = new_output.view(heads, diff, embed)
		old_output = self.output.weight.view(heads, -1, embed)

		combined_output = torch.cat((old_output, new_output), dim=1).view(heads * new_hidden, embed)
		self.output = nn.Linear(heads * hidden * proj, embed, self.bias)
		self.output.weight = nn.Parameter(combined_output)

		if self.bias:
			self.output.bias = nn.Parameter(self.output.bias.clone())

	def forward(self, x):
		# Project the input to Q, K, and V using the QKV projection layer
		qkv = self.project(x)

		# Split the heads
		qkv = qkv.view(*qkv.size()[:-1], self.heads, self.projections, -1)
		qkv = qkv.permute(0, 2, 1, 3, 4)

		return QKVEmbed(qkv.chunk(3, dim=-2))

class Attention(nn.Module):
	'''
	Attention with multiple heads, no modifications.
	'''
	
	def __init__(self,
		  hidden: int,
		  heads: int,
		  embed: Optional[int]=None,
		  mask: Optional[nn.Module]=None,
		  dropout: Optional[nn.Module]=None,
		  bias=False
		):
		'''
		Parameters
			hidden - The hidden dimension of the heads
			heads - The number of heads
			embed - The embedding size, usually tied to `hidden`
			mask - Optional mask
			dropout - Optional dropout
			bias - Whether or not to include biases, defaults to False
		'''
		super().__init__()
		
		assert hidden % heads == 0
		
		embed = default(embed, hidden)
		
		self.dropout = dropout
		self.mask = mask
		
		self.heads = heads
		self.head_dim = hidden // heads
	
	def similarity(self, q, k) -> torch.Tensor:
		sim = torch.einsum('bhid,bjd -> bhij', q, k)
		if self.mask:
			sim = sim.masked_fill(self.mask, -torch.finfo(sim.dtype).max)
		
		atn = torch.softmax(sim)
		if self.dropout:
			atn = self.dropout(atn)
		return atn
	
	def aggregate(self, qk, v) -> torch.Tensor:
		out = torch.einsum('bhij,bjd -> bhid', qk, v)
		return torch.rearrange(out, 'bhnd -> bn (hd)')
	
	def forward(self, qkv: QKVEmbed) -> torch.Tensor:
		q, k, v = qkv
		sim = self.similarity(q, k)
		return self.aggregate(sim, v)

## Span ##

class LearnableTopP(nn.Module):
	'''
	Select the top elements of a tensor based on a learnable threshold.
	'''
	
	def __init__(self, threshold, dim=-1):
		super().__init__()
		self.dim = dim
		self.threshold = nn.Parameter(torch.tensor(threshold))
	
	def forward(self, atn):
		'''
		Parameters
			atn - The attention tensor, softmax(Q @ K^T)
		'''
		
		batch_size, num_heads, seq_len, _ = atn.shape
		# Sort, sum, and mask
		atn, idx = torch.sort(atn, dim=self.dim, descending=True)
		cumsum = torch.cumsum(atn, dim=self.dim)
		mask = (cumsum <= torch.sigmoid(self.threshold))

		top_idx = torch.masked_select(idx, mask).view(batch_size, num_heads, seq_len, -1)
		
		return top_idx


class AdaptiveSpan(nn.Module):
	'''
	Learns masks which optimize for short term memory and reduces computation
	
	Adaptive Attention Span in Transformers
	https://arxiv.org/pdf/1905.07799.pdf
	'''
	
	# Stores templates for the linspace spans
	template_cache: Mapping[int, torch.Tensor] = {}
	
	def __init__(self,
		  span: int,
		  heads: int,
		  span_init: Optional[float]=None,
		  ramp: Optional[int]=None,
		  loss_co: Optional[float]=None,
		  granularity: Optional[int]=None
		):
		super().__init__()
		self.max_span = span
		self.loss_co = default(loss_co, 1e-5)
		self.heads = heads
		self.ramp = default(ramp, 32)
		self.granularity = default(granularity, 64)
		span_init = default(span_init, 0.02)
		self.span = nn.Parameter(torch.zeros(heads, 1, 1) + span_init)
	
	def forward(self, atn: torch.Tensor) -> torch.Tensor:
		return self.mask(self.trim(atn))
	
	def trim_size(self):
		'''
		Get the largest size memory is trimmed to
		'''
		
		L, GRAN = self.max_span, self.granularity
		# All heads must have the same trim length, find the largest
		max_head = math.ceil(self.span.max().item() * L)
		# span defines where mask=1 ends, ramp is linear to 0
		trim = int(min(L, max_head + self.ramp))
		return trim + GRAN - trim % GRAN
		
	def mask(self, atn: torch.Tensor) -> torch.Tensor:
		batch, seq, embed = atn.shape
		atn = atn.reshape(batch // self.heads, self.heads, seq, -1)
		
		# Get the mask template
		L = self.max_span
		if L in self.template_cache:
			template = self.template_cache[L]
		else:
			# Linearly interpolate -max_span to 0, to be shifted
			#  by span and scaled by ramp, then clamped to leave
			#  plateaus at 0 and 1 with a linear ramp between
			template = torch.linspace(1 - L, 0, L)
			self.template_cache[L] = template
		
		# Build the mask from the span
		mask = template + self.span.clamp(0, 1) * L
		mask = mask / self.ramp + 1
		mask = mask.clamp(0, 1)
		if atn.shape[-1] < self.max_span:
			mask = mask[:, :, -embed:]

		return F.normalize(atn * mask).view(batch, seq, -1)
	
	def trim(self, *m: torch.Tensor):
		"""
		Trim unnecessary memory to reduce computation
		
		Parameter
			m - Buffers to apply the same trim to
		"""
		
		trim = self.trim_size()
		return tuple(x[:, trim:, :] for x in m)

	def loss(self):
		"""A loss term for regularizing the span length"""
		
		return self.loss_co * self.max_span * self.span.mean() # + self.span.clamp(max=1) - self.span

############################
## Autoregressive wrapper ##
############################

def top_p(threshold: float = 0.9):
	'''
	Filter logits to the most likely set which exceeds the threshold.
	'''
	
	def top_p(logits):
		sorted_logits, sorted_indices = torch.sort(logits, descending=True)
		cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

		sorted_indices_to_remove = cum_probs > (1 - threshold)
		sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
		sorted_indices_to_remove[:, 0] = 0

		sorted_logits[sorted_indices_to_remove] = float('-inf')
		return sorted_logits.scatter(1, sorted_indices, sorted_logits)
	
	return top_p

def top_k(k: int):
	'''
	Only include the top k logits.
	'''
	
	def top_k(logits):
		val, ind = torch.topk(logits, k)
		probs = torch.full_like(logits, float('-inf'))
		probs.scatter_(1, ind, val)
		return probs
	
	return top_k

# Logit processor
Processor: TypeAlias = Callable[[torch.Tensor], torch.Tensor]
class GeneratePolicy:
	'''
	Describes the policy for generating text in an autoregressive language model.
	'''
	
	def __init__(self,
		processors: Optional[Processor|list[Processor]]=None,
		*,
		filter:Optional[Processor|int|float|Literal["top-k"]|Literal["top-p"]]=None,
		temperature: Optional[float]=None,
		sample: Optional[Callable[[torch.Tensor], int]]=None,
		eos_token: Optional[int]=None,
		max_len: Optional[int]=None
	):
		'''
		Parameters
			processors - A callable or list of callables which change the logit distribution
			filter - How the logits are filtered, a few options:
				callable - ordinary callable
				float [0, 1) - top-p sampling, the most likely set that exceeds the threshold
				int > 0 - top-k sampling, pick the top k most likely logits
				"top-p" - top-p with p=0.9
				"top-k" - top-k with k=24
			temperature - Temperature of the logit distribution
			sample - A function to replace the default sampling
			eos_token - End of Sequence/String/Sentence token
			max_len - Stop if no EOS before the maximum length
		'''
		
		# Set this to True to short-circuit stopping logic
		self._stop = False
		
		if processors is not None:
			if callable(processors):
				processors = [processors]
			self.processors = processors
		
		if filter is not None:
			if callable(filter):
				pass
			elif isinstance(filter, float):
				if 0 <= filter < 1:
					filter = top_p(filter)
				else:
					raise ValueError(f"top_p value must be [0, 1), got {self.filter}")
			elif isinstance(filter, int):
				filter = top_k(filter)
			elif isinstance(filter, str):
				if filter == "top-k":
					filter = top_k()
				elif filter == "top-p":
					filter = top_p()
				else:
					raise ValueError(f"Unknown filter {filter!r}")
			else:
				raise TypeError(f"Unknown filter type {type(filter)}")
			
			processors.append(filter)
		
		if temperature is not None:
			assert 0 <= temperature < 1
			processors.append(lambda x: x*temperature)
		
		if sample is not None:
			assert callable(self.sample)
			self.sample = sample
		
		if eos_token is not None:
			assert eos_token >= 0
			self.eos_token = eos_token
		
		if max_len is not None:
			assert max_len >= 0
			self.max_len = max_len
			self.count = 0
	
	def sample(self, logits):
		'''
		Default sample, softmax and multinomial
		'''
		
		P = F.softmax(logits, dim=-1)
		return torch.multinomial(P, 1)
	
	def predict(self, logits: torch.Tensor) -> int:
		'''
		All in one function which takes a logit tensor and generates a token.
		'''
		
		if hasattr(self, "processors"):
			for proc in self.processors:
				logits = proc(logits)
		
		sample =  self.sample(logits)
		if hasattr(self, "count"):
			self.count += 1
		
		if hasattr(self, "eos_token"):
			if sample == self.eos_token:
				self._stop = True
		
		return sample
	
	def stop(self):
		'''
		Default stop, checks for sequence length and EOS
		'''
		
		if self._stop:
			return True
		
		if hasattr(self, "count"):
			if self.count >= self.max_len:
				self._stop = True
				return True
		
		return False

class LanguageModel(nn.Module):
	'''
	A wrapper for an inner network which lets it act as an autoregressive
	language model. This helps because the operations involved can be fiddly.
	'''
	
	def __init__(self,
		  embed: nn.Module,
		  model: nn.Module,
		  output: nn.Module,
		  pad_token: int = 0
		):
		'''
		Parameters
			embed - The input token embedding
			model - The language model
			output - The outpu layer which outputs logits
		'''
		super().__init__()
		
		self.model = model
		
		self.pad_token = pad_token
	
	def forward(self, x: torch.Tensor, return_loss=True) -> torch.Tensor:
		'''
		The model needs to be padded properly, so use this instead of running it directly.
		'''
		
		if len(self) == 0:
			raise ValueError(f"{type(self)} has not been loaded")
		
		# Batch, Sequence
		seq = x.shape[-1]
		
		# Is the input sequence a multiple of our model dimensions?
		m = seq / self.dims
		if m.is_integer():
			# No padding needed
			padding = 0
		else:
			# Round up to the nearest multiple, then subtract the actual sequence length
			padding = int(math.ceil(m)) * self.dims - seq
			# Pad the correct side
			x = F.pad(x, (padding, 0), value=self.pad_token)
		
		if return_loss:
			# Exclude the last token from the input and first token from the output
			xi, xo = x[:, :-1], x[:, 1:]
			
			# Run the model and calculate the loss
			out = self.model(xi)
			return F.cross_entropy(out.transpose(1, 2), xo)
		
		# Slice to match the input sequence
		return self.model(x)[:, padding:]
	
	@torch.no_grad()
	def predict(self, start: torch.IntTensor, policy: Optional[GeneratePolicy]=None) -> torch.IntTensor:
		'''
		Take a Torch tensor (Sequence, Data) and return the predicted Torch tensors
		'''
		
		policy = default(policy, lambda: GeneratePolicy(filter="top-k", temperature=0.1, max_len=100))
		
		seq = start.shape[-1]
		
		out = start
		# Procedurally sample the predicted distributions
		while not policy.stop():
			# Select the most recent block
			x = out[:, -self.block_size:]
			
			logits = self.forward(x)[:, -1, :]
			token = policy.predict(logits)
			
			# Add the new prediction to the output
			out = torch.cat((out, token), dim=-1)
		
		# Get rid of the original sequence
		return out[:, seq:]
	
	def generate(self, start: str|bytes|torch.Tensor|int|Sequence[int], policy: Optional[GeneratePolicy]=None) -> bytes:
		'''
		A friendlier interface for text generation which wraps common values.
		'''
		
		if isinstance(start, str):
			start = start.encode('utf-8')
		
		if isinstance(start, bytes):
			start = torch.frombuffer(array('B', start), dtype=torch.uint8)
		
		if not isinstance(start, torch.Tensor):
			start = torch.tensor(start)
		
		num_dims = len(start.shape)
		was_training = self.training
		self.eval()
		
		# Make a 2D tensor
		if num_dims == 1:
			start = start[None, :]
		
		out = self.predict(start.type(torch.int), policy)
		
		if num_dims == 1:
			out = out.squeeze(0)
		
		self.train(was_training)
	
		# Output in bytes because the model might produce UTF-8 errors
		return bytes(out.type(torch.uint8).cpu())