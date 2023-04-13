import torch
import faiss
import numpy as np
from abc import ABC, abstractmethod
import pickle
import os

def numpy_cast(x: torch.Tensor|np.ndarray) -> np.ndarray:
	'''
	Cast input tensor into numpy array.
	'''
	if torch.is_tensor(x):
		x = x.cpu().numpy()
	return x

def torch_cast(x: np.ndarray) -> torch.Tensor:
	'''
	Cast back from numpy to torch - assumes the input is numpy.
	'''
	return torch.from_numpy(x)

class Index(ABC):
	'''
	One-way map from keys (via nearest neighbor) to indices.
	'''
	
	@abstractmethod
	def add(self, k: np.array) -> np.array:
		'''
		Add keys to the index and return the indices
		
		Parameters:
			k: Keys to add, dimensions (batch, dim)
		
		Returns: Indices of the keys, dimensions (batch,)
		'''
		pass
	
	@abstractmethod
	def search(self, q: np.array) -> np.array:
		'''
		Search for the approximate nearest neighbors of the queries and return
		the distances and indices.
		
		Parameters:
			q: Queries, dimensions (batch, dim)
		
		Returns: tuple of distances and indices, dimensions (batch, k)
		
		k is a parameter of the index and may be 1.
		'''
		pass
	
	def flush(self):
		'''
		Flushes the index to disk. By default, does nothing.
		'''
		pass

class Store(ABC):
	'''
	Stores values at indices.
	'''
	
	@abstractmethod
	def get(self, i):
		'''
		Gets the values at the indices.
		
		Parameters:
			i: Indices, dimensions (batch,)
		
		Returns: Tuple-like of columns with height `batch`. The first entry
		should be the key.
		'''
		pass
	
	@abstractmethod
	def set(self, i, v):
		'''
		Stores the values at the indices.
		
		Parameters:
			i: Indices, dimensions (batch,)
			v: Values, list-like of rows with length `batch`
		'''
		pass
	
	def flush(self):
		'''
		Flushes the store to disk. By default, does nothing.
		'''
		pass

class FaissIndex(Index):
	'''
	Index using faiss for approximate nearest neighbor search.
	'''
	
	def __init__(self, dim, path: str, k=1, *, index="hnsw", centroids=4096, probes=32, bits=8):
		self.dim = dim
		self.path = path
		self.k = k
		
		if os.path.exists(path):
			self.index = faiss.read_index(path)
		else:
			match index.lower():
				case "ivfpq":
					quantizer = faiss.IndexFlatL2(dim)
					index = faiss.IndexIVFPQ(quantizer, dim, centroids, 8, bits)
				
				case "ivfflat":
					quantizer = faiss.IndexFlatL2(dim)
					index = faiss.IndexIVFFlat(quantizer, dim, centroids)
				
				case "flat": index = faiss.IndexFlatL2(dim)
				case "pq": index = faiss.IndexPQ(dim, 8, bits)
				case "lsh": index = faiss.IndexLSH(dim, bits)
				case "hnsw": index = faiss.IndexHNSWFlat(dim, bits)
				case "hnsw2": index = faiss.IndexHNSW2Flat(dim, bits)
				
				case _: raise ValueError(f"Unknown index type {index}")
			
			index.nprobe = probes
			index.metric_type = faiss.METRIC_INNER_PRODUCT
			self.index = index
	
	def __len__(self):
		return self.index.ntotal
	
	def add(self, k):
		old = len(self)
		self.index.add(k)
		return np.arange(old, len(self))
	
	def search(self, q):
		batch, dim = q.shape
		assert dim == self.dim, f"key dim {dim} does not match index dim {self.dim}"
		
		return self.index.search(q, self.k)
	
	def flush(self):
		faiss.write_index(self.index, self.path)

class PickleStore(Store):
	'''
	Store using pickle for serialization. Probably don't use this
	except as a proof of concept.
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
		for x, i in enumerate(np.nditer(idx)):
			self.store[i.item()] = v[x]
	
	def flush(self):
		with open(self.path, "wb") as f:
			pickle.dump(self.store, f)

class Database:
	'''
	Combines an index and a store to form a key-value/metadata mapping database.
	This is the main interface which handles both numpy and torch tensors.
	Everything else in this file uses numpy.
	'''
	
	def __init__(self, index, store, training=True):
		self.index = index
		self.store = store
		self.training = training
	
	def add(self, k, v):
		k = k.reshape(-1, k.shape[-1])
		k = numpy_cast(k)
		
		v = v.reshape(-1, v.shape[-1])
		v = numpy_cast(v)
		
		i = self.index.add(k)
		self.store.set(i, v)
	
	def search(self, q):
		batch, seq, dim = q.shape
		q = q.reshape(-1, dim)
		q = numpy_cast(q)
		
		# Edge case for when the index is empty
		can_add = True
		if len(self.index) == 0:
			can_add = False
			self.add(q, q)
		
		d, i = self.index.search(q)
		if self.training and can_add:
			self.add(q, q)
		return torch_cast(self.store.get(i).reshape(batch, seq, -1, dim))
	
	def flush(self):
		self.index.flush()
		self.store.flush()

class TestDatabase(Database):
	'''
	Proof of concept database.
	'''
	
	def __init__(self, dim, path):
		super().__init__(FaissIndex(dim, path + ".index"), PickleStore(dim, path + ".store"))