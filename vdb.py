from pymilvus import connections, DataType, CollectionSchema, FieldSchema, Collection
from abc import ABC, abstractmethod
from typing import Sequence, Optional
import numpy as np
import numpy.lib.recfunctions as rcf
from dataclasses import dataclass
from common import default
import faiss
import time

def wsum(e, d):
	'''Weighted sum of embeddings e with distances d.'''
	return np.average(e, weights=np.exp(-d), axis=1)

def map_wsum(e, q, d, names):
	def combine(name):
		vector = wsum(e[name], d)
		# When all vectors are deleted (NaN), use query vector
		return np.where(np.isnan(vector), q, vector)
	
	return dict(zip(names, map(combine, names)))

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
	
	def recombine_results(self, data, q, d, tags):
		if self.recombine is None:
			return
		
		# Select only rows with at least 1 to recombine
		mask = d < self.recombine
		rows = np.any(mask, axis=1)
		data, mask, q, d = data[rows], mask[rows], q[rows], d[rows]
		
		d = np.hstack([
			np.where(mask, d, np.inf), # Mask non-recombined
			np.full((d.shape[0], 1), self.newmem) # Weighting for new memory
		])
		recids = data['id'].view(-1) # ids which were recombined
		
		# Build recombined data for insertion
		data = np.rec.fromarrays([
			data['ctime'].min(dim=1, where=mask), # Oldest creation
			data['atime'].max(dim=1, where=mask), # Newest access
			data['access'].sum(dim=1, where=mask), # All accesses
			*(wsum(np.hstack([data[name], q]), d) for name in self.VECTORS)
		], dtype=self.RowNoId)
		
		self.index.delete(recids)
		self.store.delete(recids)
		
		self.index.add(data[self.VECTORS])
		ids = self.store.insert(data)
		
		self.store.merge_tags(recids, ids)
		self.store.add_tags(ids, tags[rows])
	
	def insert_novel(self, q, d, tags):
		if self.novel is None:
			return
		
		mask = np.all(d > self.novel, axis=1) # Nothing similar
		ids = self.store.create(q[mask])
		self.store.add_tags(ids, tags[mask])
	
	def search(self, vectors, tags):
		q = vectors[self.VECTORS[0]]
		d, i = self.index.search(q, self.k) # (batch, k)
		data = self.store.get(i.view(-1)).reshape(i.shape) # (batch, k)
		
		self.recombine_results(data, q, d, tags)
		self.insert_novel(q, d, tags)
		
		# Combine top-k results
		return tuple(map_wsum(data, q, d, self.VECTORS).values())

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
		deleted = np.isin(i, self.deleted)
		return np.where(deleted, np.inf, d), np.where(deleted, 0, i)
	
	def delete(self, ids):
		if self.ndel + len(ids) > len(self.deleted):
			split = len(self.deleted) - self.ndel
			ids, extra = ids[:split], ids[split:]
			self.deleted[self.ndel:] = ids
			self.index.remove_ids(self.deleted)
			self.ndel = len(extra)
			self.deleted[:self.ndel] = extra
		else:
			ndel = self.ndel + len(ids)
			self.deleted[self.ndel: ndel] = ids
			self.ndel = ndel
	
	def save(self):
		faiss.write_index(self.index, self.path)
	
	def load(self):
		self.index = faiss.read_index(self.path)

#---- AGH

class DiscreteMemoryStore:
	def __init__(self, dim):
		self.dim = dim
		self.Row = np.dtype([
			("id", np.int64),
			("ctime", np.float64),
			("atime", np.float64),
			("access", np.int64),
			*((name, np.float32, (self.dim,)) for name in self.VECTORS)
		])
	
	def delete(self, ids):
		self.conn.execute(db.DELETE(self.PREFIX, "id in ?"), (ids,))
	
	def insert(self, data):
		cur = self.conn.executemany(
			db.INSERT(self.PREFIX, data.dtype.names, data)
		)
		return np.arange(cur.lastrowid - len(data) + 1, cur.lastrowid + 1)
	
	def create(self, **vectors):
		q = vectors[self.VECTORS[0]]
		cur = self.conn.insertmany(self.PREFIX, **vectors)
		return np.arange(cur.lastrowid - len(q) + 1, cur.lastrowid + 1)
	
	def merge_tags(self, old_ids, merged_ids):
		def combine(merge, old):
			yield merge
			yield from old
		self.conn.executemany(
			f"UPDATE {self.PREFIX}_tags SET obj = ? WHERE obj IN {', '.join('?' * old_ids.shape[1])}",
			zip(merged_ids, *old_ids)
		)
	
	def add_tags(self, ids: np.ndarray, tags: list[np.ndarray]):
		# Repeat ids for each tag
		ids = np.repeat(ids, [len(t) for t in tags])
		# Flatten tags
		tags = np.concatenate(tags)
		self.conn.executemany(
			f"INSERT INTO {self.PREFIX}_tags (obj, tag) VALUES (?, ?)",
			zip(ids, tags)
		)

class FeaturalStore:
	VECTORS = ["embedding"]
	
	def row_factory(self, row):
		id, ctime, atime, access, embedding = row
		return id, ctime, atime, access, np.frombuffer(embedding)
	
	def delete(self, ids):
		self.conn.execute(f"DELETE FROM {self.PREFIX} WHERE id in ?", (ids,))
	
	def insert(self, data):
		cur = self.conn.executemany(
			f"INSERT INTO {self.PREFIX} (ctime, atime, embedding) VALUES (?, ?, ?)",
			data[['ctime', 'atime', 'embedding']]
		)
		return np.arange(cur.lastrowid - len(data) + 1, cur.lastrowid + 1)
	
	def create(self, embedding):
		batch, dim = embedding.shapedata['ctime'].shape[:1]
		t = np.full((batch,), time.time())
		cur = self.conn.executemany(
			f"INSERT INTO {self.PREFIX} (ctime, atime, embedding) VALUES (?, ?, ?)",
			np.core.rec.fromarrays([t, t, embedding], names="ctime, atime, embedding")
		)
		return np.arange(cur.lastrowid - batch + 1, cur.lastrowid + 1)

class AssociativeStore:
	VECTORS = ["key", "value"]
	
	def row_factory(self, row):
		id, ctime, atime, access, key, value = row
		return id, ctime, atime, access, np.frombuffer(key), np.frombuffer(value)
	
	def get(self, ids):
		cur = self.conn.cursor()
		cur.row_factory = self.row_factory
		return np.array(self.conn.execute(
			f"SELECT * FROM {self.PREFIX} WHERE id in ?", (ids,)
		).fetchall(), dtype=self.Row)
	
	def delete(self, ids):
		self.conn.execute(f"DELETE FROM {self.PREFIX} WHERE id in ?", (ids,))
	
	def insert(self, data):
		cur = self.conn.executemany(
			f"INSERT INTO {self.PREFIX} (ctime, atime, embedding) VALUES (?, ?, ?)",
			data[['ctime', 'atime', 'embedding']]
		)
		return np.arange(cur.lastrowid - len(data) + 1, cur.lastrowid + 1)
	
	def create(self, embedding):
		batch, dim = embedding.shapedata['ctime'].shape[:1]
		t = np.full((batch,), time.time())
		cur = self.conn.executemany(
			f"INSERT INTO {self.PREFIX} (ctime, atime, embedding) VALUES (?, ?, ?)",
			np.core.rec.fromarrays([t, t, embedding], names="ctime, atime, embedding")
		)
		return np.arange(cur.lastrowid - batch + 1, cur.lastrowid + 1)

class DiscreteMemory:
	'''
	Base class for discrete memory layers.
	'''
	
	PREFIX = None
	VECTORS = None
	
	def __init__(self,
			index,
			store,
			k: int=1,
			recombine: Optional[float]=None,
			novel: Optional[float]=None,
			dcache: int=4096,
			newmem: float=0.0
		):
		'''
		Parameters:
			index: The approximate nearest neighbor index
			store: The vector store
			k: The number of results to return
			recombine: The distance threshold for recombining similar vectors
			novel: The distance threshold for memorizing novel vectors
			dcache: The number of deleted ids to cache before flushing
			newmem: Weight of new memory in recombination
		'''
		
		self.index = index
		self.store = store
		self.k = k
		self.recombine = recombine
		self.novel = novel
		self.newmem = newmem
	
	def search(self, vectors, tags):
		q = vectors[self.VECTORS[0]]
		d, i = self.index.search(q, self.k) # (batch, k)
		d = np.where(np.isin(i, self.deleted), np.inf, d) # Mask deletions
		data = self.store.get(i.view(-1)).reshape(i.shape) # (batch, k)
		
		if self.recombine is not None:
			mask = d < self.recombine
			rows = np.any(mask, axis=1) # Rows to recombine
			maskrows = mask[rows] # Mask rows with at least 1 to recombine
			drec = np.where(maskrows, d[rows], np.inf) # Mask non-recombined
			# Weighting for new memory
			drec = np.hstack([drec, np.full((drec.shape[0], 1), self.newmem)])
			qmask = q[rows] 
			
			rr, qr = data[rows], q[rows]
			rdat = np.rec.fromarrays([
				rr['ctime'].min(dim=1, where=maskrows), # Oldest creation
				np.full((rr.shape[0],), time.time()), # Newest access (right now)
				rr['access'].sum(dim=1, where=maskrows), # All accesses
				*(wsum(np.hstack([x, qmask]), drec, qr) for x in rr[self.VECTORS])
			], dtype=self.RowNoId)
			
			rrid = rr['id'].view(-1)
			self.index.delete(rrid)
			self.store.delete(rrid)
			ids = self.store.insert(rdat)
			self.store.merge_tags(rrid, ids)
		
		if self.novel is not None:
			mask = np.all(d > self.novel, axis=1) # Nothing similar
			ids = self.store.create(q[mask])
			self.store.add_tags(ids, tags[mask])
		
		# Combine top-k results
		return tuple(map_wsum(data, d, q, self.VECTORS).values())
	
	def delete(self, ids):
		'''
		Delete the given ids from the index using caching and masking.
		'''
		self.deleted = np.concatenate([self.deleted, ids])
		if len(self.deleted) > self.max_deleted:
			self.index.delete(self.deleted)
			self.deleted = np.zeros(0, dtype=np.int64)

class FeaturalMemory(DiscreteMemory):
	'''
	Discrete memory layer that memorizes embeddings.
	'''
	
	PREFIX = "featural"
	
	def build_row_dtype(self):
		return np.dtype([
			("id", np.int64),
			("ctime", np.float64),
			("atime", np.float64),
			("access", np.int64),
			("embedding", np.float32, (self.dim,))
		])
	
	def search(self, q, tags):
		batch, dim = q.shape
		
		data = self.masked_search(q) # (batch, k)
		m = np.frombuffer(data['embedding']).reshape(batch, -1, dim) # (batch, k, dim)
		
		if self.recombine is not None:
			R = np.argwhere(data['distance'] < self.recombine)
			dr = data[R]
			IR = dr['id']
			self.delete(IR)
			# Insert new rows
			nids = self.store.insertmany(
				'associative',
			 	embedding=wsum(m[R], dr['distance']),
				ctime=dr['ctime'].min(dim=1), # Oldest creation
				atime=dr['atime'].max(dim=1), # Newest access
				access=dr['access'].sum(dim=1) # All accesses
			)
			# Merge tags
			self.store.executemany("""
				UPDATE featural_tags SET obj_id = ?
				WHERE obj_id IN ?""", zip(nids, IR.view(-1))
			)
		
		if self.novel is not None:
			N = np.argwhere(D > self.novel)
			t = np.full_like(N, time.time(), dtype=np.float64)
			# Insert new rows
			self.store.insertmany(
				'featural',
				embedding=q[N],
				ctime=t, atime=t,
				access=np.zeros_like(N)
			)
			# Insert tags
			self.store.insertmany(
				'featural_tags',
				associative=I[N],
				tag=tags[N]
			)
		
		return wsum(m, D)

class AssociativeMemory(DiscreteMemory):
	'''
	Associative memory layer that memorizes embeddings.
	'''
	
	def __init__(self,
			dim: int,
			db,
			k: int=1,
			recombine: Optional[float]=None,
			novel: Optional[float]=None
		):
		super().__init__(dim, db, k, recombine, novel)
		self.Row = np.dtype([
			("id", np.int64),
			("ctime", np.float64),
			("atime", np.float64),
			("access", np.int64),
			("key", np.float32, (self.dim,)),
			("value", np.float32, (self.dim,))
		])
	
	def search(self, k, v, tags):
		assert k.shape == v.shape
		batch, dim = k.shape
		
		D, I = self.masked_search(k) # (batch, k)
		rows = self.store.execute(
			"SELECT * FROM associative WHERE id in ?", (I,)
		).fetchall()
		data = np.array(rows, dtype=self.Row)
		mk = np.frombuffer(data['keys']).reshape(batch, -1, dim)
		mv = np.frombuffer(data['values']).reshape(batch, -1, dim)
		
		if self.recombine is not None:
			R = np.argwhere(D < self.recombine)
			DR = D[R]
			# Delete old rows
			self.delete(I[R])
			# Insert new rows
			nids = self.store.insertmany(
				'associative',
			 	key=wsum(mk[R], DR),
				value=wsum(mv[R], DR),
				ctime=data['ctime'].min(), # Oldest creation
				atime=data['atime'].max(), # Newest access
				access=data['access'].sum() # All accesses
			)
			# Merge tags
			self.store.executemany("""
				UPDATE associative_tags SET associative_id = ?
				WHERE associative_id IN ?""", zip(nids, I[R])
			)
		
		if self.novel is not None:
			N = np.argwhere(D > self.novel)
			t = np.full_like(N, time.time(), dtype=np.float64)
			# Insert new rows
			self.store.insertmany(
				'associative',
			 	key=k[N], value=v[N],
				ctime=t, atime=t,
				access=np.zeros_like(N)
			)
			# Insert tags
			self.store.insertmany(
				'associative_tags',
				associative=I[N],
				tag=tags[N]
			)
		
		# Add the query vectors with distance 1 (max) to avoid NaN when all are deleted
		mk = np.concatenate([mk, k], axis=1)
		mv = np.concatenate([mv, v], axis=1)
		D = np.concatenate([D, np.ones((batch, 1))], axis=1)
		
		return wsum(mk, D), wsum(mv, D)

class MilvusIndex:
	def __init__(self, db, name, dim):
		self.db = db
		self.schema = CollectionSchema(
			fields=[
				FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
				FieldSchema(name='vector', dtype=DataType.FLOAT_VECTOR, dim=dim)
			]
		)
		self.collection = Collection(name=name, schema=self.schema)
	
	def search(self, q):
		return self.collection.search(
			data=[q],
			anns_field='vector',
			param={'nprobe': 16}
		)

class FaissIndex:
	def __init__(self, fname, factory, dim):
		self.index = faiss.read_index(fname)
		self.factory = factory
		self.dim = dim
	
	def search(self, q, k):
		return self.index.search(q, k)

class SqliteStore(VectorStore):
	def __init__(self, db):
		self.db = db
	
	def get(self, ids):
		return self.db.execute('SELECT vector FROM vectors WHERE id IN (?)', (ids,))

class VectorDatabase:
	def __init__(self, index, store):
		self.index = index
		self.store = store
		self.deleted = None # Table of deleted ids
	
	def search(self, q, k) -> tuple[Any, np.ndarray, np.ndarray]:
		D, I = self.index.search(q, k)
		D = np.where(np.isin(I, self.deleted), -np.inf, D)
		
		return self.store.get(I.reshape(-1)), D, I
	
	def insert(self, q, tags)
	#search, deleted, get, insert, delete