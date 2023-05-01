from pymilvus import connections, DataType, CollectionSchema, FieldSchema, Collection
from abc import ABC, abstractmethod
from typing import Sequence, Optional
import numpy as np
import numpy.lib.recfunctions as rcf
from dataclasses import dataclass
from common import default
import faiss
import time
import db
from functools import cache, cached_property, wraps

def wsum(e, d, default=None):
	'''Weighted sum of embeddings e with distances d.'''
	ws = np.average(e, weights=np.exp(-d), axis=1)
	if default is None:
		return ws
	return np.where(np.isfinite(ws), ws, default)

class DiscreteMemory:
	'''Base class for discrete memory databases.'''
	
	VECTORS = None
	
	def __init__(self,
			index,
			store,
			k: int=1,
			recombine: Optional[float]=None,
			novel: Optional[float]=None,
			newmem: float=0.0
		):
		'''
		Parameters:
			index: The approximate nearest neighbor index
			store: The vector store
			k: The number of results to return
			recombine: The distance threshold for recombining similar vectors
			novel: The distance threshold for memorizing novel vectors
			newmem: Weight of new memory in recombination
		'''
		
		self.index = index
		self.store = store
		self.k = k
		self.recombine = recombine
		self.novel = novel
		self.newmem = newmem
		
		common_rows = [
			("ctime", np.float64),
			("atime", np.float64),
			("access", np.int64),
			*((name, np.float32, (self.dim,)) for name in self.VECTORS)
		]
		
		self.Row = np.dtype([("id", np.int64), *common_rows])
		self.RowNoId = np.dtype(common_rows)
	
	def recombine_vectors(self, data, vectors, d, tags):
		if self.recombine is None:
			return
		
		# Select only rows with at least 1 to recombine
		mask = d < self.recombine
		rows = np.any(mask, axis=1)
		data, mask = data[rows], mask[rows]
		d = np.where(mask, d[rows], np.inf) # Mask non-recombined
		recids = data['id'][mask] # ids which were recombined
		
		# Build recombined data for insertion
		data = np.rec.fromarrays([
			data['ctime'].min(dim=1, where=mask), # Oldest creation
			data['atime'].max(dim=1, where=mask), # Newest access
			data['access'].sum(dim=1, where=mask), # All accesses
			*(wsum(data[n], d, v[rows]) for n, v in vectors.items())
		], dtype=self.RowNoId)
		
		# Delete the old vectors
		self.index.delete(recids)
		self.store.delete(recids)
		
		# Add the new recombined vectors
		self.index.add(data[self.VECTORS])
		ids = self.store.insert(data)
		
		# Split int a list of arrays of old ids
		recids = np.split(recids, np.cumsum(np.sum(mask, axis=1))[:-1])
		
		# Combine the tags
		self.store.merge_tags(recids, ids)
		self.store.add_tags(ids, tags[rows])
	
	def insert_novel(self, vectors, d, tags):
		if self.novel is None:
			return
		
		mask = np.all(d > self.novel, axis=1) # Nothing similar
		ids = self.store.create({n: v[mask] for n, v in vectors.item()})
		self.store.add_tags(ids, tags[mask])
	
	def search(self, tags, q, **vectors):
		d, i = self.index.search(q, self.k) # (batch, k)
		data = self.store.get(i.reshape(-1)).reshape(i.shape)
		
		self.recombine_vectors(data, vectors, d, tags)
		self.insert_novel(vectors, d, tags)
		
		# Combine top-k results
		return tuple(wsum(data[n], d, v) for n, v in vectors.items())

class FeaturalMemory(DiscreteMemory):
	VECTORS = ["embedding"]
	
	def search(self, q, tags):
		return super().search(tags, q, embedding=q)

class AssociativeMemory(DiscreteMemory):
	VECTORS = ["key", "value"]
	
	def search(self, k, v, tags):
		return super().search(tags, k, key=k, value=v)

class DMFaissIndex:
	'''Faiss index for discrete memory.'''
	
	def __init__(self, dim, path):
		self.dim = dim
		self.index = faiss.IndexFlatIP(dim)
		self.path = path
		self.deleted = np.empty((4096,), dtype=np.int64)
		self.ndel = 0
	
	def add(self, vectors):
		assert len(vectors.shape) == 2, f"Expected 2D array, got {vectors.shape}"
		assert vectors.shape[1] == self.dim, f"Expected {self.dim}, got {vectors.shape[1]}"
		
		self.index.add(vectors)
	
	def search(self, vectors, k=1):
		assert len(vectors.shape) == 2, f"Expected 2D array, got {vectors.shape}"
		assert vectors.shape[1] == self.dim, f"Expected {self.dim}, got {vectors.shape[1]}"
		
		d, i = self.index.search(vectors, k)
		deleted = np.isin(i, self.deleted) # Mask deletions
		return np.where(deleted, np.inf, d), np.where(deleted, 0, i + 1)
	
	def delete(self, ids):
		# Cache overflow, commit deletions
		if self.ndel + len(ids) > len(self.deleted):
			split = len(self.deleted) - self.ndel
			ids, extra = ids[:split], ids[split:]
			self.deleted[self.ndel:] = ids
			self.index.remove_ids(self.deleted - 1)
			# Store remainder
			self.ndel = len(extra)
			self.deleted[:self.ndel] = extra
		else:
			ndel = self.ndel + len(ids)
			self.deleted[self.ndel: ndel] = ids
			self.ndel = ndel
	
	def commit(self):
		self.index.remove_ids(self.deleted[:self.ndel])
		faiss.write_index(self.index, self.path)
		self.ndel = 0
	
	def load(self):
		self.index = faiss.read_index(self.path)

class DMSqliteStore:
	'''Sqlite store for discrete memory.'''
	
	def __init__(self, dim, path):
		self.dim = dim
		self.conn = db.connect(path)
		self.conn.row_factory = self.row_factory
		self.main_table = prefix
		self.tags_table = f"{prefix}_tags"
		self.conn.executescript(f"""
			CREATE TABLE IF NOT EXISTS {self.PREFIX} (
				id INTEGER PRIMARY KEY,
				ctime REAL NOT NULL,
				atime REAL NOT NULL,
				access INTEGER NOT NULL,
				{', '.join(f"{name} BLOB NOT NULL" for name in self.VECTORS)}
			);
			INSERT INTO {self.PREFIX} (id, ctime, atime, access) VALUES (0, 0, 0, 0);
			CREATE TABLE IF NOT EXISTS {self.PREFIX}_tags (
				obj INTEGER NOT NULL,
				tag INTEGER NOT NULL,
				PRIMARY KEY (obj, tag)
			);
		""")
	
	def row_factory(self, row):
		id, ctime, atime, access, *vectors = row
		return (id, ctime, atime, access, *map(np.frombuffer, vectors))
	
	def get(self, ids):
		cur = self.conn.cursor()
		cur.row_factory = self.row_factory
		return np.array(self.conn.execute(
			db.SELECT("*", self.main_table, db.IN('id', len(ids))), ids
		).fetchall(), dtype=self.Row)
	
	def delete(self, ids):
		self.conn.execute(db.DELETE(self.main_table, db.IN('id', len(ids))), ids)
	
	def insert(self, data):
		cur = self.conn.executemany(db.INSERT(self.main_table, data.dtype.names), data)
		return np.arange(cur.lastrowid - len(data) + 1, cur.lastrowid + 1)
	
	def create(self, vectors: dict[str, np.ndarray]):
		return self.insert(np.rec.fromarrays(
			vectors.values(), dtype=[(n, np.float64, self.dim) for n in vectors]
		))
	
	def merge_tags(self, old: list[list[int]], new: list[int]):
		for ids, nid, tag in zip(old, new, tag):
			objin = db.IN('obj', len(ids))
			
			self.conn.execute(f"""
				INSERT INTO {self.tags_table} (obj, tag, count)
					SELECT ?, tag, SUM(count)
					FROM tags WHERE {objin} GROUP BY tag
			""", (nid, *ids))
			self.conn.execute(
				f"DELETE FROM {self.tags_table} WHERE {objin}", ids
			)
		self.conn.commit()
	
	def add_tags(self, ids: list[int], tags: list[list[str]]):
		tt = self.tags_table
		for obj, ntags in zip(ids, tags):
			self.conn.execute(f"""
				INSERT OR REPLACE INTO {tt} (obj, tag, count)
					SELECT ?, id, COALESCE({tt}.count, 0) + 1 FROM tags
					LEFT JOIN {tt} ON {tt}.obj = ? AND {tt}.tag = tags.id
					WHERE {db.IN('name', len(tags))}
			""", (obj, *ntags))
		self.conn.commit()