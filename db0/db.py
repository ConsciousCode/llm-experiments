'''
Abstract definitions for database components
'''

import torch
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple, Optional, Any

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

def is_normalized(x: np.ndarray) -> bool:
	'''
	Checks if the input is normalized.
	'''
	return np.allclose(np.linalg.norm(x, axis=-1), 1)

class VectorIndex(ABC):
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
	
	def commit(self):
		'''
		Commit any changes. By default, does nothing.
		'''
		pass

class VectorStore(ABC):
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
	
	def commit(self):
		'''
		Commit any changes. By default, does nothing.
		'''
		pass

class VectorDatabase(ABC):
	'''
	Base class for databases mapping vectors to data.
	'''
	
	@abstractmethod
	def add(self, k: np.ndarray, v: Any):
		'''
		Associate a key vector with a value.
		'''
		pass
	
	@abstractmethod
	def search(self, q: np.ndarray, k=1) -> Any:
		'''
		Search for the top-k values associated with the query vector.
		'''
		pass
	
	def commit(self):
		'''
		Commit any changes. By default, does nothing.
		'''
		pass

class SimpleDatabase:
	'''
	Combines an index and a store to form a key-value/metadata mapping database.
	This is the main interface which handles both numpy and torch tensors.
	Everything else should use numpy.
	'''
	
	def __init__(self, index, store, training=True, pressure=0.5):
		self.index = index
		self.store = store
		self.training = training
		self.pressure = pressure
	
	def add(self, k, v):
		k, v = numpy_cast(k), numpy_cast(v)
		k = k.reshape(-1, k.shape[-1])
		v = v.reshape(-1, v.shape[-1])
		assert is_normalized(k), "Key vectors must be normalized"
		assert is_normalized(v), "Value vectors must be normalized"
		
		*seq, dim = k.shape
		print("Adding", k.size // dim, "keys")
		
		i = self.index.add(k)
		self.store.set(i, v)
		
		print("Total", len(self.index))
	
	def search(self, q):
		batch, seq, dim = q.shape
		q = numpy_cast(q)
		q = q.reshape(-1, q.shape[-1])
		assert is_normalized(q), f"Query vectors must be normalized"
		
		# Edge case for when the index is empty
		can_add = True
		if len(self.index) == 0:
			can_add = False
			self.add(q, q)
		
		d, i = self.index.search(q)
		if self.training and can_add:
			print("distances", d)
			print("any positive?", np.any(d > 0))
			q = q[np.where(-d > self.pressure)[0],:]
			print("new", q.shape)
			self.add(q, q)
		return torch_cast(self.store.get(i).reshape(batch, seq, -1, dim))
	
	def commit(self):
		self.index.commit()
		self.store.commit()

@dataclass
class TopK:
	'''
	Top-K results, allows for both SoA and AoS access
	'''
	k: int
	distance: np.ndarray
	vector: np.ndarray
	idx: np.ndarray
	
	class Index(NamedTuple):
		'''SoA access to a single result'''
		distance: float
		vector: np.ndarray
		idx: int
	
	def __iter__(self): return iter(self.idx)
	def __len__(self): return self.k
	def __getitem__(self, i):
		return TopK.Index(self.distance[i], self.vector[i], self.idx[i])

def sanitize_name(name):
	# Be noisy, don't silently correct just in case it isn't sanitized later
	if not name.isidentifier():
		raise ValueError(f"Invalid name {name!r}")
	return name.lower()