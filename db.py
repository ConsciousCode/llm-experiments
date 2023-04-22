from dataclasses import dataclass, field
from typing import Optional
import time
import sqlite3
import re

def schema(schema):
	'''
	Metadata for database schema fields.
	'''
	
	required = not re.search("(?i)(autoincrement|default)", schema)
	foreign = re.search("(?i)foreign", schema)
	
	schema = re.sub(r"(?i)(foreign|autoincrement)", "", schema)
	schema = re.sub(r"\s+", " ", schema.strip())
	
	return field(metadata={"db": {
		"schema": schema,
		"foreign": foreign,
		"required": required
	}})

def isrequired(field):
	return field.metadata.get("db", {}).get("required", False)

@dataclass
class Memory:
	'''
	Core memory stream.
	'''
	
	id: int = schema("INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT")
	msg: str = schema("TEXT NOT NULL")
	ctime: float = schema("TIMESTAMP NOT NULL")
	atime: float = schema("TIMESTAMP")
	importance: Optional[int] = schema("INTEGER")
	
	def __str__(self):
		ts = time.strftime("%H:%M:%S", time.localtime(self.ctime))
		return f"{ts} {self.msg}\n"

@dataclass
class Debug:
	'''
	Debug messages.
	'''
	
	id: int = schema("INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT")
	msg: str = schema("TEXT NOT NULL")
	time: Optional[float] = schema("TIMESTAMP NOT NULL")

class Table:
	'''
	SQL database table to more easily interact with the database.
	'''
	
	def __init__(self, schema, conn):
		self.conn = conn
		self.cursor = conn.cursor()
		self.cursor.row_factory = lambda cur, row: schema(*row)
		
		self.schema = schema
		self.name = schema.__name__
		self.fields = schema.__dataclass_fields__
		self.required = tuple(k for k, v in self.fields.items() if isrequired(v))
		
		self.cursor.execute(f"CREATE TABLE IF NOT EXISTS {self!s}")
		self.commit()
	
	def __str__(self):
		fields = []
		for k, v in self.fields.items():
			db = v.metadata.get("db", {})
			if db.get("foreign", False):
				k = f"FOREIGN KEY({k})"
			fields.append(f"{k} {db['schema']}")
		return f"{self.name} ({', '.join(fields)})"
	
	def __repr__(self): return f"Table({self.name!r})"
	def __iter__(self): return iter(self.cursor)
	def fetchone(self): return self.cursor.fetchone()
	def fetchall(self): return self.cursor.fetchall()
	@property
	def lastrowid(self): return self.cursor.lastrowid
	
	def select(self, **kwargs):
		preds = ' AND '.join(f"{k} = ?" for k in kwargs)
		return self.conn.execute(
			f"SELECT * FROM {self.name} WHERE {preds}", kwargs.values()
		)
	
	def insert(self, **kwargs):
		k = ', '.join(kwargs.keys())
		v = ', '.join('?'*len(kwargs))
		self.cursor.execute(
			f"INSERT INTO {self.name} ({k}) VALUES ({v})", tuple(kwargs.values())
		)
		return self
	
	def concat(self, col, value, **kwargs):
		preds = ' AND '.join(f"{k} = ?" for k in kwargs)
		self.cursor.execute(
			f"UPDATE {self.name} SET {col} = {col} || ? WHERE {preds}", (value, *kwargs.values())
		)
		return self
	
	def count(self):
		return self.cursor.execute(f"SELECT COUNT(*) FROM {self.name}").fetchone()[0]
	
	def commit(self):
		self.conn.commit()
		return self

class Database:
	def __init__(self, path):
		self.path = path
		self.conn = sqlite3.connect(path)
		self.conn.row_factory = sqlite3.Row
		
		self.memory = Table(Memory, self.conn)
		self.debug = Table(Debug, self.conn)
	
	def execute(self, sql, *params):
		return self.conn.execute(sql, *params)
	
	def commit(self):
		self.conn.commit()
		return self
	
	def recent(self, lines):
		recent = self.conn.execute(
			f"SELECT * FROM memory ORDER BY ctime DESC LIMIT {lines}"
		).fetchall()[::-1]
		
		for row in recent:
			yield Memory(*row)