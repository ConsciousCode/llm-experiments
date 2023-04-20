'''
Implement an index using faiss directly for approximate nearest neighbor search.
'''

from . import Index
import faiss
import numpy as np
import os

## NOTE: Not currently used

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