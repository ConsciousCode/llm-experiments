import numpy as np
import faiss
import os
import sqlite3
from typing import Optional
from functools import cache

TABLE = "CREATE TABLE IF NOT EXISTS"
INT = "INTEGER NOT NULL"
ID = f"id {INT} PRIMARY KEY AUTOINCREMENT"
TIME = f"{INT} DEFAULT (strftime('%s', 'now'))"
SCHEMA = f"""
{TABLE} tags (
	id {ID},
	name TEXT NOT NULL UNIQUE
);
{TABLE} implicit (
	id {ID},
	ctime {TIME},
	atime {TIME},
	access {INT} DEFAULT 0,
	deleted {INT} DEFAULT 0,
	key BLOB NOT NULL,
	value BLOB NOT NULL
);
{TABLE} tagmap (
	obj {INT} REFERENCES associative(id),
	tag {INT} REFERENCES tags(id),
	PRIMARY KEY (obj, tag),
	UNIQUE (obj, tag)
)
"""

@cache
def LIST(count: int) -> str:
	return f"({', '.join('?' * count)})"

def wsum(e, d, default=None):
	'''Weighted sum of embeddings e with distances d.'''
	ws = np.average(e, weights=np.exp(-d), axis=1)
	if default is None:
		return ws
	return np.where(np.isfinite(ws), ws, default)

class AssociativeMemory:
	'''Combined index and store for discrete associative memory.'''
	
	def __init__(self,
			index,
			store,
			k: int=1,
			recombine: Optional[float]=None,
			novel: Optional[float]=None
		):
		'''
		Parameters:
			index: The approximate nearest neighbor index
			store: The vector store
			k: The number of results to return
			recombine: The distance threshold for recombining similar vectors
			novel: The distance threshold for memorizing novel vectors
		'''
		
		self.index = index
		self.store = store
		self.k = k
		self.recombine = recombine
		self.novel = novel
		
		self.Row = np.dtype([
			("id", np.uint64),
			("ctime", np.uint64),
			("atime", np.uint64),
			("access", np.uint64),
			('deleted', np.uint64),
			("key", np.float32, (self.dim,)),
			("value", np.float32, (self.dim,))
		])
	
	def recombine_vectors(self, data, keys, values, d, tags):
		if self.recombine is None:
			return
		
		# Select only rows with at least 1 to recombine
		mask = d < self.recombine
		rows = np.any(mask, axis=1)
		data, mask = data[rows], mask[rows]
		d = np.where(mask, d[rows], np.inf) # Mask non-recombined
		recids = data['id'][mask] # ids which were recombined
		
		wk = wsum(data['key'], d, keys[rows])
		wv = wsum(data['value'], d, values[rows])
		
		# Delete the old vectors
		self.index.delete(recids)
		self.store.delete(recids)
		
		# Add the new recombined vectors
		self.index.add(wk)
		ids = self.store.insert(
			ctime=data['ctime'].min(dim=1, where=mask), # Oldest creation
			atime=data['atime'].max(dim=1, where=mask), # Newest access
			access=data['access'].sum(dim=1, where=mask), # All accesses
			key=wk, value=wv
		)
		
		# Split into a list of arrays of old ids
		recids = np.split(recids, np.cumsum(np.sum(mask, axis=1))[:-1])
		
		# Combine the tags
		self.store.merge_tags(recids, ids)
		self.store.add_tags(ids, tags.mask(rows))
	
	def insert_novel(self, keys, values, d, tags):
		if self.novel is None:
			return
		
		mask = (d > self.novel) & np.isfinite(d)
		rows = np.all(mask, axis=1) # Nothing similar
		ids = self.store.create(keys[rows], values[rows])
		self.store.add_tags(ids, tags.mask(rows))
	
	def search(self, keys, values, tags):
		d, i = self.index.search(keys, self.k) # (batch, k)
		data = self.store.get(i.reshape(-1))
		data = np.array(data, dtype=self.Row).reshape(i.shape)
		d = np.where(data['deleted'] == 1, np.inf, d)
		
		self.recombine_vectors(data, keys, values, d, tags)
		self.insert_novel(keys, values, d, tags)
		
		# Combine top-k 
		keys = wsum(data['key'], d, keys)
		values = wsum(data['value'], d, values)
		return keys, values

class FaissIndex:
	'''Faiss index for discrete memory.'''
	
	def __init__(self, dim, path, factory="Flat"):
		self.dim = dim
		self.path = path
		if os.path.exists(path):
			self.load()
		else:
			self.index = faiss.index_factory(dim, factory)
	
	def add(self, keys):
		self.buffer.add(keys)
	
	def delete(self, keys):
		'''Does nothing (other indexes might need to delete).'''
		pass
	
	def search(self, keys, k=1):
		return self.index.search(keys, k)
	
	def commit(self):
		faiss.write_index(self.index, self.path)
	
	def load(self):
		self.index = faiss.read_index(self.path)

def tobytes(x):
	for row in x:
		yield row.tobytes()

def implicit_row_factory(row):
	id, ctime, atime, access, key, value = row
	return id, ctime, atime, access, np.frombuffer(key), np.frombuffer(value)

class SqliteStore:
	'''Sqlite store for discrete memory.'''
	
	def __init__(self, path):
		self.conn = sqlite3.connect(path)
		self.conn.row_factory = implicit_row_factory
	
	def get(self, ids):
		# Update access information
		self.conn.executemany("""
			UPDATE implicit SET
				atime = strftime('%s', 'now'),
				access = access + 1
			WHERE id = ?
		""", ids)
		self.conn.commit()
		# Query data
		return self.conn.executemany(
			"SELECT * FROM implicit WHERE id = ?", ids
		).fetchall()
	
	def delete(self, ids):
		self.conn.executemany(
			"UPDATE implicit SET deleted = 1 WHERE id = ?", ids
		)
		self.conn.commit()
	
	def insert(self, ctime, atime, access, key, value):
		cur = self.conn.executemany("""
			INSERT INTO implicit (ctime, atime, access, key, value)
			VALUES (?, ?, ?, ?, ?)
		""", zip(ctime, atime, access, tobytes(key), tobytes(value)))
		self.conn.commit()
		return np.arange(cur.lastrowid - len(ctime) + 1, cur.lastrowid + 1)
	
	def create(self, keys, values):
		cur = self.conn.executemany(
			"INSERT INTO implicit (key, value) VALUES (?, ?)",
			zip(tobytes(keys), tobytes(values))
		)
		self.conn.commit()
		return np.arange(cur.lastrowid - len(keys) + 1, cur.lastrowid + 1)
	
	def merge_tags(self, old, new):
		for ids, nid in zip(old, new):
			ids = list(ids)
			
			# Need to query using IN to get the COUNT
			self.conn.execute(f"""
				INSERT INTO tagmap (obj, tag, count)
				SELECT {nid}, tag, SUM(count)
				FROM tags WHERE obj IN {LIST(len(ids))} GROUP BY tag
			""", ids)
			self.conn.executemany(
				"DELETE FROM tagmap WHERE obj = ?", ids
			)
		self.conn.commit()
	
	def add_tags(self, ids, tags):
		for obj, ntags in zip(ids, tags):
			self.conn.execute(f"""
				INSERT OR REPLACE INTO tagmap (obj, tag, count)
				SELECT ?1, id, COALESCE(tagmap.count, 0) + 1 FROM tags
				LEFT JOIN tagmap ON tagmap.obj = ?1 AND tagmap.tag = tags.id
				WHERE name = ?2
			""", ((obj, tag) for tag in ntags))
		self.conn.commit()
	
	def commit(self):
		self.conn.commit()