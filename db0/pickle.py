'''
Implement a store using pickle for serialization. Probably don't use this
except as a proof of concept.
'''

from . import Store
import numpy as np
import os
import pickle

## NOTE: Not currently used

class PickleStore(Store):
	'''
	Store using pickle for serialization.
	'''
	
	def __init__(self, dim, path):
		self.dim = dim
		self.path = path
		if os.path.exists(path):
			with open(self.path, "rb") as f:
				self.store = pickle.load(f)
		else:
			self.store = {}
	
	def __len__(self):
		return len(self.store)
	
	def get(self, idx):
		w, h = idx.shape
		result = np.empty((w, h, self.dim), dtype=np.float32)
		for i in range(w):
			for j in range(h):
				result[i, j] = self.store[idx[i, j]]
		return result
	
	def set(self, idx, v):
		# idx: (batch,)
		# v: (batch, dim)
		if idx.size == 0:
			return
		
		for x, i in enumerate(np.nditer(idx)):
			self.store[i.item()] = v[x]
	
	def commit(self):
		with open(self.path, "wb") as f:
			pickle.dump(self.store, f)