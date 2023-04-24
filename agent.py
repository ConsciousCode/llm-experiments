from dataclasses import dataclass, field
from typing import Optional, Iterator
import time

import prompt
import db
from itertools import chain
import re
import json
import printf

from complete import complete

NAME = "Orin"
CHAT_LINES = 25
DB_FILE = "memory.db"
PRESSURE = 1/2
EMOTION = None#"disoriented"

LEVEL = prompt.LOG_LEVEL.index("debug")

class StreamCapture:
	'''
	Convert a list of patterns into a regex pattern that matches every prefix
	of the total pattern, then applies this to a stream of tokens.
	
	Example:
		XYZ\s+=\s+(\d+)?\s* becomes
		(?:X(?:Y(?:Z(?:\s+(?:=(?:\s+(?:((?P<END>\d+)))?)?)?)?)
	'''
	def __init__(self, pattern):
		# Add inner parentheses to the last pattern to detect final match
		pattern = [*pattern[:-1], f"(?P<END>{pattern[-1]})"]
		pattern = "".join(f"(?:{p}" for p in pattern) + '?'.join(')' * len(pattern))
		self.pattern = re.compile(pattern)
	
	def capture(self, stream):
		buf = ""
		m = None
		stream = iter(stream)
		for token in stream:
			buf += token
			if m := self.pattern.search(buf):
				# Stop searching if there can't be more
				if m.end() < len(buf):
					if m.lastgroup == "END": break
					yield m[0]
					buf = buf[m.end():]
			else:
				yield token
			
		if m:
			# Yield either the match or its text if it's not full
			yield m if m.lastgroup == "END" else m[0]
			if buf := buf[m.end():]:
				yield buf
		yield from stream

IMPORTANCE_PARAM = StreamCapture([
	*prompt.IMPORTANCE_TAG, "\s+", "=", "\s+", "(\d+)"
])
EMOTION_PARAM = StreamCapture([
	*prompt.EMOTION_TAG, "\s+", "=", "\s+",
	'"', "((?:(?:[^\\\\]+|\\\\.)*)+)", '"'
])

def tee(stream: Iterator) -> tuple[Iterator, list]:
	'''Tee a stream into a list.'''
	# Note to self: Stop deleting this! Just keep it around in case we
	#  need it again.
	def impl(stream, arr):
		for item in stream:
			arr.append(item)
			yield item
	arr = []
	return impl(stream, arr), arr

@dataclass
class Message:
	origin: db.Origin
	message: Optional[str] = None
	ctime: float = field(default_factory=time.time)
	importance: Optional[int] = None
	
	def __str__(self):
		return f"{prompt.timestamp(self.ctime)} {self.output()}"
	
	@staticmethod
	def from_explicit(explicit: db.ExplicitMemory):
		'''Convert an explicit message to a user message'''
		return Message(explicit.origin, explicit.message, explicit.ctime, explicit.importance)
	
	def to_explicit(self) -> db.ExplicitMemory:
		'''Convert a user message to an explicit message'''
		return db.ExplicitMemory(None,
			ctime=self.ctime, atime=self.ctime, access=0,
			origin=self.origin, message=self.message, importance=self.importance
		)
	
	def format_name(self):
		if self.origin.name in {"SYSTEM", "SUMMARY"}:
			return f"[{self.origin.name}]"
		return f"<{self.origin.name}>"
	
	def output(self):
		'''Output as viewed by the user'''
		out = self.format_name()
		if self.message is not None:
			out += f" {self.message}"
		return out

class Agent:
	def __init__(self):
		self.db = db.Database(DB_FILE)
		self.chatlog = []
		
		self.db.state.load_defaults(
			name = NAME,
			loglevel = LEVEL,
			pressure = PRESSURE,
			emotion = EMOTION,
			unsummarized = 0,
			lines = CHAT_LINES,
		)
		# Set during debugging so we don't summarize a bunch of reboots
		self.db.state.unsummarized = 0
		
		self.debug(f"__init__({self.name!r})")
		
		self.reload()
		self.add_memory(self.message("SYSTEM", prompt.reload()))
	
	@property
	def name(self): return self.db.state.name
	@property
	def loglevel(self): return self.db.state.loglevel
	@property
	def pressure(self): return self.db.state.pressure
	@property
	def emotion(self): return self.db.state.emotion
	@property
	def unsummarized(self): return self.db.state.unsummarized
	@property
	def lines(self): return self.db.state.lines
	
	def message(self, origin, msg=None, importance=None) -> Message:
		'''Create a message from a string'''
		return Message(self.db.origin(origin), msg, time.time(), importance)
	
	def log(self, level: int, msg: str):
		'''Debug print'''
		
		self.db.log(level, msg)
		
		if level <= self.loglevel:
			printf.log(level, msg)
	
	def error(self, msg): self.log(1, msg)
	def warn(self, msg): self.log(2, msg)
	def info(self, msg): self.log(3, msg)
	def debug(self, msg): self.log(4, msg)
	def verbose(self, msg): self.log(5, msg)
	
	def command(self, user: str, cmd: str, args: list[str]):
		'''Process a command.'''
		
		self.debug(f"command({user!r}, {cmd!r}, {args!r})")
		
		match cmd.lower():
			case "select"|"update"|"insert"|"delete":
				yield from self.command(user, "sql", [cmd.upper(), *args])
			case "sql":
				try:
					rows = self.db.execute(' '.join(args))
					self.db.commit()
					print(rows)
					printf.json([dict(row) for row in rows.fetchall()])
				except Exception as e:
					self.error(f"SQL ERROR: {e}")
			case "state":
				if len(args) == 0:
					printf.json(self.db.state.cache)
				elif len(args) == 1:
					printf.json(self.db.state[args[0]])
				else:
					try:
						value = json.loads(args[1])
					except json.JSONDecodeError:
						value = args[1]
					self.db.state[args[0]] = value
					self.info(f"Set state[{args[0]!r}] = {value!r}")
			case "sum"|"summary"|"summarize":
				yield from self.summarize(self.chatlog)
				yield '\n'
			case "prompt":
				yield self.build_prompt() + "\n"
			case "level":
				if len(args) > 0:
					level = args[0].upper()
					if level in prompt.LOG_LEVEL:
						il = prompt.LOG_LEVEL.index(level)
					else:
						try:
							il = int(level)
						except ValueError:
							self.error("Invalid log level")
							return
					
					self.db.state.loglevel = il
					self.info(f"level = {level} ({il})")
				else:
					il = self.loglevel
					if il > len(prompt.LOG_LEVEL):
						level = "verbose" + "+" * (il - len(prompt.LOG_LEVEL))
					else:
						level = prompt.LOG_LEVEL[il]
					yield f"level = {level} ({il})\n"
			case _:
				self.error(f"Unknown command {cmd}")
	
	def build_prompt(self):
		self.debug("build_prompt()")
		return prompt.master(self.name, self.emotion) + '\n'.join(self.chatlog)
	
	def reload(self):
		'''Reload the agent's most recent memories.'''
		
		self.debug("reload()")
		self.chatlog = [str(Message.from_explicit(msg)) for msg in self.db.recent(self.lines)]
	
	def add_message(self, msg: str):
		'''Add a chat message to the chat log.'''
		
		self.debug(f"add_message({msg!r})")
		self.db.state.unsummarized += 1
		print("Update", self.db.state.unsummarized)
		
		# Rolling chat log
		self.chatlog = self.chatlog[-CHAT_LINES:] + [msg]
	
	def add_memory(self, msg: Message) -> int:
		'''Add a memory to the database, which also adds to chat log.'''
		
		yield '\n'
		self.debug(f"add_memory({msg!r})")
		
		# All memories are added to the internal chat log
		self.add_message(str(msg))
		return self.db.insert_explicit(msg.to_explicit())
	
	def complete(self, prompt: str):
		'''AI prompt completion.'''
		
		self.verbose(f"complete({prompt!r})")
		return complete(prompt)
	
	def summarize(self, dialog: str):
		'''Summarize the current chat log.'''
		
		self.verbose(f"summarize({dialog!r})")
		
		self.db.state.unsummarized = 0
		importance = None
		summary = []
		
		# Get the summary completion
		stream = self.complete(prompt.summarize(self.name, dialog))
		for token in IMPORTANCE_PARAM.capture(stream):
			if isinstance(token, str):
				yield token
				summary.append(token)
			else:
				importance = token[1]
		
		# Update the importance
		if importance:
			print("Importance =", importance)
			try:
				importance = int(importance[0], 10)
			except ValueError as e:
				self.log(f"Invalid importance: {e}")
				importance = None
		
		self.add_memory(self.message(prompt.SUMMARY_TAG, ''.join(summary), importance=importance))
	
	def chat(self, user: str, msg: str) -> Iterator[str]:
		'''Respond to the user's message.'''
		
		self.verbose(f"chat({user!r}, {msg!r})")
		
		# Add user message
		self.add_memory(self.message(user, msg))
		
		# Process AI message
		pending = self.message(self.name)
		po = pending.output()
		
		completion = []
		p = f"{self.build_prompt()}\n{po}"
		stream = self.complete(p)
		
		yield po
		
		for token in EMOTION_PARAM.capture(stream):
			if isinstance(token, str):
				yield token
				completion.append(token)
			else:
				self.db.state.emotion = token[1]
		
		pending.message = ''.join(completion)
		self.add_memory(pending)
		
		# Summarize every so often
		if self.unsummarized >= self.lines * self.pressure:
			yield from self.summarize(self.build_prompt())