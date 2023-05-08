#!/usr/bin/env python3
'''
Distill a parent model to a clone with its FF layers replaced with discrete
memory layers.
'''

import os
import lmkd
import model
from functools import cached_property
from llm.client import GRPCModel
from transformers.models.auto import AutoConfig, AutoModelForCausalLM, AutoTokenizer

TEACHER_NAME = "hf-internal-testing/tiny-random-gptj" #"databricks/dolly-v1-6b"
DATASET_NAME = "ag_news"

DEFAULT_K = 5
RECOMBINE = 2/3
NOVEL = 1/3
TEMPERATURE = 1.0
PORT = os.path.abspath("llm.sock")

class LMKD(lmkd.Distill):
	def __init__(self,
	    	debug=None,
		    device=None,
		    *,
			teacher_name=TEACHER_NAME,
			dataset_name=DATASET_NAME,
			k=DEFAULT_K,
			recombine=RECOMBINE,
			novel=NOVEL,
			temperature=TEMPERATURE,
			port=PORT
		):
		super().__init__(debug, device)
		
		self.teacher_name = teacher_name
		self.dataset_name = dataset_name
		self.k = k
		self.recombine = recombine
		self.novel = novel
		self.temperature = temperature
		self.port = port
	
	@cached_property
	def config(self):
		return AutoConfig.from_pretrained(self.teacher_name)
	
	@cached_property
	def batch_size(self):
		return self.config.n_positions
	
	def teacher(self):
		return GRPCModel(self.port)
	
	def student(self, state_dict=None):
		orin = model.build_gptj(self.config)
		print(orin)
		if state_dict is None:
			teacher = AutoModelForCausalLM.from_pretrained(self.teacher_name)
			state_dict = model.clone_gptj(teacher)
		
		# Sanity check, there should be no unused keys
		assert set(state_dict.keys()).issubset(orin.state_dict().keys())
		
		# strict=False to allow for missing keys
		orin.load_state_dict(state_dict, strict=False)
		return orin
	
	def tokenizer(self):
		tokenizer = AutoTokenizer.from_pretrained(self.teacher_name)
		return lambda text: tokenizer(text, return_tensors="pt")
	
	def dataset(self, split):
		return self.dataset_name

if __name__ == "__main__":
	lmkd.main()