from typing import Iterator
import time
import sqlite3
import json

from dataclasses import dataclass
from functools import cache
from typing import Optional, Mapping

from collections.abc import MutableMapping

# Reduce repetition
TABLE = "CREATE TABLE IF NOT EXISTS"
INT = "INTEGER NOT NULL"
TEXT = "TEXT NOT NULL"
IDENT = f"id {INT} PRIMARY KEY AUTOINCREMENT"
FNOW = f"{INT} DEFAULT (strftime('%s', 'now'))"
TIMER = f"ctime {FNOW}, atime {FNOW}, access {INT} DEFAULT 0"
MEMORY = f"{IDENT}, {TIMER}"
# Database schema
SCHEMA = f"""
{TABLE} log ({IDENT},
	time {INT},
	level {INT},
	message {TEXT}
);
{TABLE} state ( -- Basically a JSON object
	key {TEXT} PRIMARY KEY,
	value {TEXT}
);
{TABLE} origins ({IDENT},
	name {TEXT} UNIQUE
);
{TABLE} explicit ({MEMORY},
	origin {INT} REFERENCES origins(id),
	message {TEXT},
	--embedding BLOB NOT NULL,
	importance INTEGER
);
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
def LIST(count: int) -> str:
	return f"({', '.join('?' * count)})"

@cache
def INSERT(table: str, fields: tuple[str, ...]) -> str:
	values = LIST(len(fields))
	fields = ', '.join(map(sanitize, fields))
	return f"INSERT INTO {sanitize(table)} ({fields}) VALUES {values}"

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

@cache
def IN(name: str, count: int) -> str:
	return f"{name} IN {LIST(count)}"

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

class StateProxy(MutableMapping):
	'''Proxy for the state table.'''
	
	def __init__(self, conn):
		# Store connection not db to avoid ref loops
		self._conn = conn
		self._cache = {}
	
	def get(self, k: str, default=None):
		'''Get a value from the state dict or an optional default.'''
		
		if k in self._cache:
			return self._cache[k]
		cur = self._conn.execute("SELECT value FROM state WHERE key = ?", (k,))
		if row := cur.fetchone():
			self._cache[k] = val = json.loads(row[0])
			return val
		return default
	
	def update(self, mapping: Mapping[str, object]):
		'''Mapping update.'''
		
		self._cache.update(mapping)
		self._conn.executemany(
			"INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)",
			mapping.items()
		)
		self._conn.commit()
	
	def reload(self):
		'''Reload the cache.'''
		
		self._cache.clear()
		cur = self._conn.execute("SELECT key, value FROM state")
		for k, v in cur:
			self._cache[k] = json.loads(v)
	
	def load_defaults(self, **kwargs):
		'''Insert a default value if the key is not present.'''
		
		self._conn.executemany(
			"INSERT OR IGNORE INTO state (key, value) VALUES (?, ?)",
			((k, json.dumps(v)) for k, v in kwargs.items())
		)
		self._conn.commit()
		self.reload()
	
	def __contains__(self, k: str):
		return self.get(k, ...) is ...
	
	def __getitem__(self, k: str):
		return self.get(k)
	
	def __setitem__(self, x: str, y):
		self._cache[x] = y
		self._conn.execute(
			"INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)",
			(x, json.dumps(y))
		)
		self._conn.commit()
	
	def __getattr__(self, name: str):
		if name.startswith("_"):
			return super().__getattr__(name)
		return self[name]
	
	def __setattr__(self, name: str, value):
		if name.startswith("_"):
			return super().__setattr__(name, value)
		self[name] = value
	
	def __repr__(self):
		return repr(self._cache)

class Database:
	def __init__(self, conn):
		if isinstance(conn, str):
			conn = sqlite3.connect(conn)
		self.conn = conn
		self.conn.row_factory = sqlite3.Row
		self.conn.executescript(SCHEMA)
		self.conn.commit()
		self.origins = {}
		self.state = StateProxy(self.conn)
	
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
				row = self.conn.execute(
					"SELECT id FROM origins WHERE name = ?", (ident,)
				).fetchone()
				if row:
					id = row[0]
				else:
					# Brand new origin
					cur = self.conn.execute(
						"INSERT INTO origins (name) VALUES (?)", (ident,)
					)
					self.commit()
					id = cur.lastrowid
			else:
				id = origin
		elif isinstance(id := ident, int):
			if origin is None:
				row = self.conn.execute(
					"SELECT name FROM origins WHERE id = ?", (ident,)
				).fetchone()
				if row:
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
		
		recent = self.conn.execute(
			f"SELECT * FROM explicit ORDER BY ctime DESC LIMIT {lines}"
		).fetchall()[::-1]
		
		for row in recent:
			yield ExplicitMemory(**row, origin=self.origin(row.origin))
	
	def log(self, level: int, msg: str) -> int:
		'''Insert a new log entry. Returns the id.'''
		cur = self.conn.execute(
			"INSERT INTO log (time, level, message) VALUES (?, ?, ?)",
			(time.time(), level, msg)
		)
		self.conn.commit()
		return cur.lastrowid
	
	def insert_explicit(self, memory: ExplicitMemory) -> int:
		'''Insert an explicit memory. Returns the id.'''
		cur = self.conn.execute(
			"INSERT INTO explicit (origin, message, importance) VALUES (?, ?, ?)",
			(memory.origin.id, memory.message, memory.importance)
		)
		self.conn.commit()
		return cur.lastrowid
