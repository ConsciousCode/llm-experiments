#!/usr/bin/env python3

import torch.nn as nn
from transformers.models.gptj.configuration_gptj import GPTJConfig
import tf
from dataclasses import dataclass

class DMTransformerBlock(nn.Module):
	def __init__(self, dmem, attn):
		super().__init__()
		self.dmem = dmem
		self.attn = attn
	
	def forward(self, x, ctx):
		x = self.dmem(x, ctx)
		x = self.attn(x, ctx)
		return x

class TransformersLMWrapper(tf.LanguageModel):
	'''
	Wrapper converting transformers-style inputs to InfoStill-style inputs.
	'''
	
	def forward(self, *, input_ids, **kwargs):
		return self.super().forward(input_ids, **kwargs)

@dataclass
class OrinConfig:
	embed: int
	layers: int
	heads: int
	max_seq: int
	dropout_p: float
	pdrop_attn: float
	prenorm_epsilon: float

def build_layers(config):
	def mha(selector):
		return tf.MultiheadAttention(
			config.embed, config.heads,
			selector=selector, rotary_embed=True
		)
	
	for layer in range(config.layers):
		yield DMTransformerBlock(
			mha(tf.ScaledDotProductSelector(
				config.max_seq, dropout=config.pdrop_attn
			)),
			mha(tf.AssociativeMemorySelector(
				config.embed, max_seq=config.max_seq, dropout=config.pdrop_attn
			))
		)

def build(config):
	return tf.Residual(list(build_layers(config)))

def build_gptj(config: GPTJConfig):
	'''
	Builds a model from a given config.
	'''
	
	config = OrinConfig(
		embed=config.n_embd,
		layers=config.n_layer,
		heads=config.n_head,
		max_seq=config.n_positions,
		dropout_p=config.resid_pdrop,
		pdrop_attn=config.attn_pdrop,
		prenorm_epsilon=config.layer_norm_epsilon
	)
	
	return TransformersLMWrapper(
		**config.asdict()
	)