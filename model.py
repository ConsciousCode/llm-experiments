#!/usr/bin/env python3
'''
Classes and functions for building the Orin model.
'''

import torch
import torch.nn as nn
from transformers.models.gptj.configuration_gptj import GPTJConfig
import tf
from dataclasses import dataclass
from collections import OrderedDict
import re
import vdb
import os

class Memory:
	'''
	Associative memory for storing discrete memories.
	'''
	
	def __init__(self, dim, path, k, recombine, novel):
		index_path = os.path.join(path, ".index.db")
		store_path = os.path.join(path, ".store.db")
		self.memory = vdb.AssociativeMemory(
			vdb.FaissIndex(dim, index_path, factory="Flat"),
			vdb.SqliteStore(store_path),
			k, recombine, novel
		)
	
	def search(self, keys, values, ctx):
		return self.memory.search(keys, values, ctx.tags)

class DMTransformerBlock(nn.Module):
	'''
	Basic unit of discrete memory transformer.
	'''
	
	def __init__(self, config):
		super().__init__()
		self.prenorm = nn.LayerNorm(config.embed, eps=config.prenorm)
		self.attn = tf.MultiheadAttention(
			config.embed, config.heads, rotary_embed=True,
			selector=tf.ScaledDotProductSelector(
				config.max_seq, dropout=config.pdrop_attn
			),
			bias=False
		)
		self.dmem = tf.MultiheadAttention(
			config.embed, config.heads, rotary_embed=False,
			selector=tf.AssociativeMemorySelector(
				config.embed, max_seq=config.max_seq, dropout=config.pdrop_attn
			),
			bias=False
		)
	
	def forward(self, x, ctx):
		x = self.prenorm(x)
		x = self.attn(x, ctx)
		x = self.dmem(x, ctx)
		return x

class TransformersLMWrapper(tf.LanguageModel):
	'''
	Wrapper converting transformers-style inputs to InfoStill-style inputs.
	'''
	
	def forward(self, *, input_ids, **kwargs):
		return self.super().forward(input_ids, **kwargs)

@dataclass
class OrinConfig:
	vocab: int
	embed: int
	layers: int
	heads: int
	max_seq: int
	dropout_p: float
	pdrop_attn: float
	prenorm: float
	postnorm: float

def build_layers(config):
	for layer in range(config.layers):
		yield DMTransformerBlock(config)

def build(config):
	return tf.Residual(list(build_layers(config)))

def build_gptj(config: GPTJConfig):
	'''
	Builds a model from a given config.
	'''
	
	config = OrinConfig(
		vocab=config.vocab_size,
		embed=config.n_embd,
		layers=config.n_layer,
		heads=config.n_head,
		max_seq=config.n_positions,
		dropout_p=config.resid_pdrop,
		pdrop_attn=config.attn_pdrop,
		prenorm=config.layer_norm_epsilon,
		postnorm=config.layer_norm_epsilon
	)
	
	return TransformersLMWrapper(
		vocab=config.vocab,
		embed=config.embed,
		model=build(config),
		dropout=config.dropout_p,
		postnorm=config.postnorm
	)

def gptj_to_orin(key):
	# Dropouts not in state_dict
	#
	# drop -> dropout
	# h -> lm
	# wte -> embed
	# h.*.attn -> attn
	#   bias -> selector.bias
	#   masked_bias -> None (unused even in GPT-J)
	#   ln_1 -> prenorm
	#   attn_dropout -> selector.attn_dropout
	#   resid_dropout -> resid_dropout
	#   {q, k, v}_proj -> qkv_proj
	#   out_proj -> out_proj
	# None -> dmem
	# h.*.mlp.* -> None
	# ln_f -> postnorm
	
	if "mlp" in key or "masked_bias" in key:
		return None
	
	key = re.sub(r"^h", "lm", key)
	key = key.replace("wte", "embed")
	key = key.replace("attn.bias", "attn.selector.bias")
	key = key.replace("ln_1", "prenorm")
	key = key.replace("ln_f", "postnorm")
	
	return key

def clone_gptj(parent):
	parent = parent.state_dict()
	state = OrderedDict()
	skipped = set()
	proj = set()
	
	print("Parent state_dict:", parent.keys())
	
	# Weight renaming
	for tk, tw in parent.items():
		sk = gptj_to_orin(tk)
		if sk is None:
			skipped.add(tk)
			continue
		
		# Don't add {q, k, v}_proj, to be combined later
		if m := re.match("^(.+?\.)[^.]+(?<!out)_proj(.*)$", tk):
			proj.add(f"{m[1]}*{m[2]}")
		else:
			state[sk] = tw
	
	print("Skipped keys:", skipped)
	
	# Combine QKV
	for p in proj:
		qkv_proj = torch.cat([parent[p.replace("*", f"{s}_proj")] for s in "qkv"], dim=0)
		state[gptj_to_orin(p.replace("*", "qkv_proj"))] = qkv_proj
	
	print("Student state_dict:", state.keys())
	
	return state