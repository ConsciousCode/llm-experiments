from dataclasses import dataclass, field
from typing import Optional, Iterator
import time
import sqlite3
import re
from functools import cache
import json

TABLE = "CREATE TABLE IF NOT EXISTS"
IDENT = "id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT"
TIMER = "ctime TIMESTAMP NOT NULL, atime TIMESTAMP NOT NULL, access INTEGER NOT NULL DEFAULT 0"
MEMORY = f"{IDENT}, {TIMER}"
SCHEMA = f"""
{TABLE} log ({IDENT},
	time TIMESTAMP NOT NULL,
	level INTEGER NOT NULL,
	message TEXT NOT NULL
);
{TABLE} state (
	key TEXT NOT NULL PRIMARY KEY,
	value TEXT NOT NULL
);
{TABLE} origins ({IDENT},
	name TEXT NOT NULL
);
{TABLE} explicit ({MEMORY},
	origin INTEGER REFERENCES origins(id) NOT NULL,
	message TEXT NOT NULL,
	--embedding BLOB NOT NULL,
	importance INTEGER
);
{TABLE} featural ({MEMORY},
	embedding BLOB NOT NULL
);
{TABLE} associative ({MEMORY},
	key BLOB NOT NULL,
	value BLOB NOT NULL
)
"""
@cache
def INSERT(table, fields):
	'''Shorthand for INSERT statement'''
	fields = fields.split(' ')
	return f"INSERT INTO {table} ({', '.join(fields)}) VALUES ({', '.join('?' * len(fields))})"

@dataclass
class Identified:
	id: int

@dataclass
class MemoryEntry(Identified):
	'''Purposefully nasty name to avoid instantiation'''
	ctime: float
	atime: float
	access: int

@dataclass
class Log(Identified):
	time: float
	level: int
	message: str

@dataclass
class Origin(Identified):
	name: str

@dataclass
class ExplicitMemory(MemoryEntry):
	origin: Origin
	message: str
	importance: Optional[int]

@dataclass
class FeaturalMemory(MemoryEntry):
	embedding: bytes

@dataclass
class AssociativeMemory(MemoryEntry):
	key: bytes
	value: bytes

class StateProxy:
	'''Proxy for the state table.'''
	def __init__(self, conn):
		self.conn = conn
		self.cache = {}
	
	def get(self, k: str, default=...):
		if k in self.cache:
			return self.cache[k]
		cur = self.conn.execute("SELECT value FROM state WHERE key = ?", (k,))
		if row := cur.fetchone():
			val = json.loads(row[0])
			self.cache[k] = val
			return val
		if default is ...:
			raise KeyError(k)
		return default
	
	def reload(self):
		'''Reload the cache.'''
		self.cache.clear()
		cur = self.conn.execute("SELECT key, value FROM state")
		for k, v in cur:
			self.cache[k] = json.loads(v)
	
	def load_defaults(self, **kwargs):
		'''Insert a default value if the key is not present.'''
		self.conn.executemany(
			"INSERT OR IGNORE INTO state (key, value) VALUES (?, ?)",
			[(k, json.dumps(v)) for k, v in kwargs.items()]
		)
		self.conn.commit()
		self.reload()
	
	def __contains__(self, k: str):
		MISSING = object()
		return k in self.cache or self.get(k, MISSING) is MISSING
	
	def __getitem__(self, k: str):
		v = self.get(k)
		if v == ...:
			raise KeyError(k)
		return v
	
	def __setitem__(self, x: str, y):
		self.cache[x] = y
		self.conn.execute(
			"INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)",
			(x, json.dumps(y))
		)
		self.conn.commit()
	
	def __getattr__(self, name: str):
		if name in {'conn', 'cache'}:
			return super().__getattr__(name)
		return self[name]
	
	def __setattr__(self, name: str, value):
		if name in {'conn', 'cache'}:
			return super().__setattr__(name, value)
		self[name] = value
	
	def __repr__(self):
		return repr(self.cache)

class Database:
	def __init__(self, path):
		self.path = path
		self.conn = sqlite3.connect(path)
		self.conn.row_factory = sqlite3.Row
		self.conn.executescript(SCHEMA)
		self.conn.commit()
		self.origins = {}
		self.state = StateProxy(self.conn)
	
	def execute(self, sql, *params):
		return self.conn.execute(sql, *params)
	
	def commit(self):
		self.conn.commit()
		return self
	
	def _insert_lastrowid(self, table, fields, values):
		cur = self.execute(INSERT(table, fields), values)
		self.conn.commit()
		return cur.lastrowid
	
	def origin(self, ident: int|str|Origin) -> Origin:
		'''
		Convert an id or name to an origin, may insert a new one.
		'''
		if isinstance(ident, Origin):
			return ident
		other = self.origins.get(ident, None)
		if isinstance(ident, str):
			name, id = ident, other
			if id is None:
				cur = self.execute("SELECT id FROM origins WHERE name = ?", (ident,))
				if row := cur.fetchone():
					id = row[0]
				else:
					id = self._insert_lastrowid("origins", "name", (ident,))
				name = ident
		elif isinstance(ident, int):
			id, name = ident, other
			if name is None:
				cur = self.execute("SELECT name FROM origins WHERE id = ?", (ident,))
				if row := cur.fetchone():
					name = row[0]
				else:
					raise ValueError("No such origin")
				id = ident
		else:
			raise TypeError("origin must be str or int")
		self.origins[name] = id
		self.origins[id] = name
		return Origin(id, name)
	
	def recent(self, lines: int) -> Iterator[ExplicitMemory]:
		recent = self.execute(
			f"SELECT * FROM explicit ORDER BY ctime DESC LIMIT {int(lines)}"
		).fetchall()[::-1]
		
		for row in recent:
			yield ExplicitMemory(*row[:4], self.origin(row[4]), *row[5:])
	
	def log(self, level: int, msg: str) -> int:
		return self._insert_lastrowid(
			"log", "time level message", (time.time(), level, msg)
		)
	
	def insert_explicit(self, memory: ExplicitMemory) -> int:
		origin = self.origin(memory.origin).id
		return self._insert_lastrowid(
			"explicit", "ctime atime origin message importance",
			(memory.ctime, memory.atime, origin, memory.message, memory.importance)
		)