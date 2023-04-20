import sqlite3
import sqlite_vss
import faiss
from dataclasses import dataclass
import numpy as np
from typing import NamedTuple, Optional

INSERT_VECTOR = "INSERT INTO {table} (vector) VALUES (?)"
INSERT_DATA = "INSERT INTO {table} ({keys}) VALUES ({values})"
LAST_ROW_ID = "SELECT id FROM {table} ORDER BY id DESC LIMIT 1"
SELECT_TOPK = "SELECT rowId, vector FROM {table} WHERE vss_search(vector, ?) LIMIT ?"

@dataclass
class EmbedTable:
	'''
	Local cache and understanding of an vector table
	'''
	id: int
	name: str
	dim: int
	schema: Optional[tuple[str]]
	data_cache: Optional[list[tuple]]
	vector_cache: faiss.IndexFlatIP
	lastrowid: Optional[int]

class SqliteDatabase:
	'''
	Wrapper class around sqlite and sqlite_vss which adds faiss indexing.
	This adds a layer of caching to the database, which is flushed to the
	database when the cache reaches a certain size to ensure that insertions
	are batched and there are enough vectors to train a new index.
	'''
	
	def __init__(self, path, buffer=4096):
		self.path = path
		self.db = sqlite3.connect(path)
		self.db.enable_load_extension(True)
		sqlite_vss.load(self.db)
		self.db.row_factory = sqlite3.Row
		self.cursor = self.db.cursor()
		
		self.buffer = buffer
		
		with open("schema.sql", "r") as f:
			self.execute(f.read())
		
		# Local copy of vector tables and their caches
		# Caching is necessary to keep insertions batched and to gather enough
		#  vectors to train a new index
		self.vtab = {}
		for k, v in schema.tables.items():
			k = sanitize_name(k)
			lastrowid = self.query(
				LAST_ROW_ID.format(table=k)
			).fetchone()
			if lastrowid is not None:
				lastrowid = lastrowid['id']
			
			dc = None
			if cols := v.schema:
				cols = tuple(cols.keys())
				dc = []
			
			vtab = EmbedTable(k, v.dim, cols, dc, faiss.IndexFlatIP(v.dim), lastrowid)
			self.vtab[k] = vtab
	
	def query(self, query, params=None):
		'''Execute an arbitrary SQL query with parameters - does not commit'''
		return self.execute(query, params, commit=False)
	
	def execute(self, query, params=None, commit=True):
		'''Execute an arbitrary SQL query with parameters'''
		result = self.cursor.execute(query, params or ())
		if commit:
			self.commit()
		return result
	
	def executemany(self, query, params=None, commit=True):
		'''Execute an arbitrary SQL query with multiple instances of parameters'''
		if params:
			result = self.cursor.executemany(query, params)
			if commit:
				self.commit()
			return result
	
	def commit(self):
		'''Commit the database'''
		return self.cursor.commit()
	
	def flush(self, name=None):
		'''Flush the local vector cache to the database'''
		
		if name is None:
			for name in self.vtab:
				self.flush(name)
		else:
			name = sanitize_name(name)
			vtab = self.vtab[name]
			
			# Insert the vector
			vector = vtab.vector_cache.reconstruct_n(0, vtab.ntotal)
			query = INSERT_VECTOR.format(table=name)
			self.execute(query, (vector,), commit=False)
			
			# If there's data associated with the vector
			if dc := vtab.data_cache:
				query = INSERT_DATA.format(
					table=name,
					keys=', '.join(vtab.schema),
					values=', '.join('?'*len(vtab.schema))
				)
				self.executemany(query, dc)
				dc.clear()
			
			vtab.vector_cache.reset()
	
	def vector_insert(self, name: str, vectors: np.ndarray|list[np.ndarray], data: Optional[dict|list[dict]]=None):
		'''
		Insert vectors into the database
		
		Parameters:
			name: The vector table to insert into
			vectors: The vectors to insert (D or NxD)
		'''
		
		if isinstance(vectors, list):
			vectors = np.vstack(vectors)
		
		name = sanitize_name(name)
		dim = vectors.shape[-1]
		
		assert len(vectors.shape) <= 2, f"Expected 1D or 2D array, got {vectors.shape}"
		assert dim == vtab.dim, f"Expected {vtab.dim}, got {dim} for vector table {name!r} insert"
		
		vectors = vectors.reshape(-1, dim)
		n = vectors.shape[0]
		
		vtab = self.vtab[name]
		
		# Add to data cache (convert to column tuples)
		if vtab.data_cache is not None:
			if isinstance(data, dict):
				data = [data]
			cols = vtab.schema
			vtab.data_cache.extend(tuple(d[c] for c in cols) for d in data)
		
		# Add to local cache and maybe flush
		vtab.vector_cache.add(vectors)
		if vtab.vector_cache.ntotal > self.buffer:
			self.flush()
			self.lastrowid = self.lastrowid
		elif self.lastrowid is None:
			self.lastrowid = n - 1
		else:
			self.lastrowid += n
		
		# Return new ids
		return np.arange(self.lastrowid - n + 1, self.lastrowid)
	
	def vector_search(self, name: str, query: np.ndarray, k=1) -> TopK:
		'''
		Search for the top k vectors in the database
		
		Parameters:
			name: The vector table to search
			query: The vector to search for
			k: The number of results to return
		'''
		
		name = sanitize_name(name)
		vtab = self.vtab[name]
		
		assert len(query.shape) == 1, f"Expected 1D array, got {query.shape}"
		assert query.shape[0] == vtab.dim, f"Expected {vtab.dim}, got {query.shape[0]} for vector table {name!r} search"
		
		# SQL top-k
		rows = self.execute(SELECT_TOPK.format(table=name), (query, k))
		sql_idx, vectors = [], []
		for row in rows.fetchall():
			sql_idx.append(row["rowId"])
			vectors.append(row["vector"])
		
		# Local top-k
		_, cache_idx = vtab.search(query, k)
		cache_idx = cache_idx.flatten()
		vectors.extend(vtab.reconstruct(i) for i in cache_idx)
		vectors = np.vstack(vectors)
		
		# Combine results
		index = faiss.IndexFlatIP(vtab.dim)
		index.add(vectors)
		D, topk = index.search(query, k)
		
		# Truncate to top-k
		topk = topk[:k]
		
		idx = np.concatenate((sql_idx, cache_idx))
		return TopK(k, D, vectors[topk], idx[topk])
	
	def vector_query(self, name: str, query: np.ndarray, k=1) -> 