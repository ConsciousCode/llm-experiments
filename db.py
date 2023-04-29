from dataclasses import dataclass
from typing import Optional, Iterator
import time
import sqlite3
from functools import cache
import json
import numpy as np

# Reduce repetition
TABLE = "CREATE TABLE IF NOT EXISTS"
IDENT = "id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT"
FNOW = "INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))"
TIMER = f"ctime {FNOW}, atime {FNOW}, access INTEGER NOT NULL DEFAULT 0"
MEMORY = f"{IDENT}, {TIMER}"

# Database schema
SCHEMA = f"""
{TABLE} log ({IDENT},
	time TIMESTAMP NOT NULL,
	level INTEGER NOT NULL,
	message TEXT NOT NULL
);
{TABLE} state ( -- Basically a JSON object
	key TEXT NOT NULL PRIMARY KEY,
	value TEXT NOT NULL
);
{TABLE} origins ({IDENT},
	name TEXT NOT NULL UNIQUE
);
{TABLE} tags ({IDENT},
	name TEXT NOT NULL UNIQUE
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
);
{TABLE} associative_tags(
	obj INTEGER REFERENCES associative(id) NOT NULL,
	tag INTEGER REFERENCES tags(id) NOT NULL,
	PRIMARY KEY (obj, tag),
	UNIQUE (obj, tag)
)
"""

@cache
def sanitize(name: str) -> str:
	'''Sanitize an identifier, raise an error if it's invalid'''
	
	if not isinstance(name, str):
		raise TypeError(f"Expected str, got {type(name).__name__}")
	if not name.isidentifier():
		raise ValueError(f"Invalid identifier: {name}")
	return name.lower()

@cache
def INSERT(table: str, fields: tuple[str, ...]) -> str:
	values = ', '.join("?" * len(fields))
	fields = ', '.join(map(sanitize, fields))
	return f"INSERT INTO {sanitize(table)} ({fields}) VALUES ({values})"

@cache
def SELECT(col: str|tuple[str, ...], table: str, fields: tuple[str, ...]) -> str:
	if isinstance(col, str):
		col = (sanitize(col),)
	col = ', '.join(map(sanitize, col))
	pred = ' AND '.join(f"{sanitize(k)} = ?" for k in fields)
	return f"SELECT {col} FROM {sanitize(table)} WHERE {pred}"

@cache
def DELETE(table: str, where: Optional[str]=None) -> str:
	s = f"DELETE FROM {sanitize(table)}"
	return s if where is None else f"{s} WHERE {where}"

@cache
def UPDATE(table: str, fields: tuple[str, ...], where: Optional[str]=None) -> str:
	fields = ', '.join(f"{sanitize(k)} = ?" for k in fields)
	s = f"UPDATE {sanitize(table)} SET {fields}"
	return s if where is None else f"{s} WHERE {where}"

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
		# Store connection not db to avoid ref loops
		self.conn = conn
		self.cache = {}
	
	def get(self, k: str, default=...):
		'''Get a value from the state dict or an optional default.'''
		
		if k in self.cache:
			return self.cache[k]
		cur = self.conn.execute(SELECT('value', 'state', key=k))
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
		return self.get(k)
	
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
	
	# Generic SQL methods
	
	def execute(self, sql, *params):
		return self.conn.execute(sql, *params)
	
	def select(self, col, table, **kwargs):
		'''Shorthand for select statement.'''
		return self.execute(SELECT(col, table, tuple(kwargs.keys())), tuple(kwargs.values()))
	
	def insert(self, table, **kwargs):
		'''Shorthand for insert statement.'''
		cur = self.execute(INSERT(table, tuple(kwargs.keys())), tuple(kwargs.values()))
		self.conn.commit()
		return cur
	
	def commit(self):
		'''Commit the current transaction.'''
		self.conn.commit()
		return self
	
	# Specific database methods
	
	def origin(self, ident: int|str|Origin) -> Origin:
		'''Convert an id or name to an origin, may insert a new one.'''
		# Check if the origin needs work
		if isinstance(ident, Origin):
			if ident.id is None:
				ident = origin.name
			elif ident.name is None:
				ident = origin.id
			else:
				return ident
		
		# Figure out what kind of origin we have
		origin = self.origins.get(ident, None)
		if isinstance(name := ident, str):
			if origin is None:
				row = self.select('id', 'origins', name=ident).fetchone()
				id = row[0] if row else self.insert("origins", name=ident).lastrowid
			else:
				id = origin
		elif isinstance(id := ident, int):
			if origin is None:
				if row := self.select('name', 'origins', id=ident).fetchone():
					name = row[0]
				else:
					raise ValueError("No such origin")
			else:
				name = origin
		else:
			raise TypeError("origin must be str|int|Origin(str|int)")
		
		# Update the cache and return
		self.origins[name] = id
		self.origins[id] = name
		return Origin(id, name)
	
	def recent(self, lines: int) -> Iterator[ExplicitMemory]:
		'''Get the `lines` most recent explicit memories.'''
		# Sanitization
		if not isinstance(lines, int):
			raise TypeError("lines must be int")
		
		recent = self.execute(
			f"SELECT * FROM explicit ORDER BY ctime DESC LIMIT {lines}"
		).fetchall()[::-1]
		
		for row in recent:
			yield ExplicitMemory(
				row['id'], row['ctime'], row['atime'], row['access'],
				self.origin(row['origin']), row['message'], row['importance']
			)
	
	def log(self, level: int, msg: str) -> int:
		'''Insert a new log entry. Returns the id.'''
		return self.insert("log",
			time=time.time(), level=level, message=msg
		).lastrowid
	
	def insert_explicit(self, memory: ExplicitMemory) -> int:
		'''Insert an explicit memory. Returns the id.'''
		return self.insert("explicit",
			ctime=memory.ctime, atime=memory.atime, access=memory.access,
			origin=self.origin(memory.origin).id,
			message=memory.message,
			importance=memory.importance
		).lastrowid
