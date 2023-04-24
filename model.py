#!/usr/bin/env python3

import torch.nn as nn
from typing import Optional
from transformers.models.gptj.configuration_gptj import GPTJConfig

from common import NORM, DROP, default
from tf import Residual, FeaturalMemory, AssociativeMemory, ScaledDotProductSelector, MultiheadAttention, LanguageModel

class DMTransformerBlock(nn.Module):
	def __init__(self, dmem, attn):
		super().__init__()
		self.dmem = dmem
		self.attn = attn
	
	def forward(self, x, ctx):
		x = self.dmem(x, ctx)
		x = self.attn(x, ctx)
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
		
		def block(layer, rotary=False):
			if layer < layers // 2:
				memory = FeaturalMemory(embed)
			else:
				memory = AssociativeMemory(embed, max_seq=max_seq)
			
			return DMTransformerBlock(
				MultiheadAttention(embed, heads,
					selector=ScaledDotProductSelector(
						max_seq, dropout=pdrop_attn
					),
					rotary_embed=rotary
				),
				memory
			)
		
		self.layers = Residual([
			block(0, rotary=True),
			*(block(i) for i in range(1, layers))
		])
	
	def forward(self, x, ctx):
		return self.layers(x, ctx)

class TransformersWrapper(nn.Module):
	'''
	Wrapper converting transformers-style inputs to InfoStill-style inputs.
	'''
	
	def __init__(self, model):
		super().__init__()
		
		self.model = model
	
	def forward(self, *, input_ids, **kwargs):
		return self.model(input_ids, **kwargs)

def build_gptj(config: GPTJConfig):
	'''
	Builds a model from a given config.
	'''
	
	model = DMTransformer(
		embed=1,#config.n_embd,
		layers=1,#config.n_layer,
		heads=1,#config.n_head,
		max_seq=1,#config.n_positions,
		
		dropout_p=1,#config.resid_pdrop,
		pdrop_attn=1,#config.attn_pdrop,
		
		prenorm_epsilon=1,#config.layer_norm_epsilon
	)
	
	model = LanguageModel(
		vocab=1,#config.vocab_size,
		embed=1,#config.n_embd,
		max_seq=1,#config.n_position,
		model=model,
		dropout=1,#config.embd_pdrop,
		postnorm=1,#config.layer_norm_epsilon
	)
	
	return TransformersWrapper(model)