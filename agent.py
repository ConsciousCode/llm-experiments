from diatree import Diatree
from dataclasses import dataclass, field
import openai
import os
from typing import Optional, Iterator
import time
import pprint

import prompt
import db
import llm.client as client
from itertools import chain

openai.api_key = os.environ.get("API_KEY", None)

DEBUG = True
NAME = "Ziggy"
LLM_ENGINE = "text-davinci-003"
CHAT_LINES = 25
DB_FILE = "memory.db"
PRESSURE = 1/2

def complete_chatgpt(prompt, *, engine=LLM_ENGINE, max_tokens=100, temperature=0.9, top_k=0, top_p=0.9, frequency_penalty=0.0, presence_penalty=0.0, stop=None):
	stop = stop or []
	stop.append("\n")
	
	response = openai.Completion.create(
		engine=engine,
		prompt=prompt,
		max_tokens=max_tokens,
		temperature=temperature,
		top_k=top_k,
		top_p=top_p,
		frequency_penalty=frequency_penalty,
		presence_penalty=presence_penalty,
		stop=stop,
		stream=True
	)
	
	for token in response:
		yield token.choices[0].text

def complete_local(prompt, *, max_tokens=100, temperature=0.9, top_k=0, top_p=0.9, frequency_penalty=0.0, presence_penalty=0.0, stop=None):
	stop = stop or []
	stop.append("\n")
	
	response = client.complete(
		prompt=prompt,
		max_tokens=max_tokens,
		temperature=temperature,
		top_k=top_k,
		top_p=top_p,
		frequency_penalty=frequency_penalty,
		presence_penalty=presence_penalty,
		stop=stop
	)
	
	yield from response

complete = complete_local

@dataclass
class Message:
	name: str
	time: float = field(default_factory=time.time)
	
	def __str__(self):
		t = time.strftime("%H:%M:%S", time.localtime(self.time))
		return f"{t} {self.onlyname()}"
	
	def onlyname(self):
		return f"<{self.name}>"

def tee_impl(it, arr):
	'''Tee an iterator into a list and return both.'''
	for x in it:
		arr.append(x)
		yield x

def tee(it):
	'''Tee an iterator into a list and return both.'''
	arr = []
	return tee_impl(it, arr), arr

class Agent:
	def __init__(self):
		self.name = NAME
		self.db = db.Database(DB_FILE)
		self.debug = DEBUG
		self.unsummarized = 0
		self.lines = CHAT_LINES
		
		self.dprint(f"__init__({self.name!r})")
		
		self.reload()
		self.add_memory(prompt.reload())
	
	def dprint(self, msg):
		'''Debug print'''
		
		#print(msg)
		self.db.debug.insert(msg=msg, time=time.time()).commit()
	
	def reload(self):
		'''Reload the agent's most recent memories.'''
		
		self.dprint("reload()")
		self.reprompt(f"\n{msg}" for msg in self.db.recent(self.lines))
	
	def reprompt(self, dialog):
		'''Rebuild the dialog tree for prompting.'''
		
		self.dialog = Diatree(prompt.master(self.name), dialog)
	
	def command(self, user: str, cmd: str, args: list[str]):
		'''Process a command.'''
		
		self.dprint(f"command({user!r}, {cmd!r}, {args!r})")
		
		match cmd:
			case "sql":
				try:
					rows = self.db.execute(' '.join(args))
					self.db.commit()
					yield pprint.pformat([dict(row) for row in rows.fetchall()])
					yield "\n"
				except Exception as e:
					self.dprint(f"SQL ERROR:\n{e}")
					yield str(e)
			case "sum"|"summary"|"summarize":
				summary = f"[SUMMARY] {self.summarize()}"
				self.add_memory(summary)
				yield summary
			case "dia"|"diatree":
				yield from self.dialog
				yield "\n"
			case "debug":
				self.debug = not self.debug
				yield f"Debug set to {self.debug}"
			case _:
				self.dprint(f"Unknown command {cmd}")
				yield "Unknown command"
	
	def add_message(self, msg: str):
		'''Add a chat message to the chat log.'''
		
		self.dprint(f"add_message({msg!r})")
		
		# Rolling chat log
		self.reprompt(self.dialog[1:][-CHAT_LINES:] + msg)
	
	def add_memory(self, msg: str) -> int:
		'''Add a memory to the database, which also adds to chat log.'''
		
		self.dprint(f"add_memory({msg!r})")
		
		ctime = time.time()
		
		# All memories are added to the internal chat log
		self.unsummarized += 1
		self.add_message(f"\n{prompt.timestamp(ctime)} {msg}")
		return self.db.memory.insert(msg=msg, ctime=ctime).commit().lastrowid
	
	def stream(self, memory: str, id: Optional[int]=None):
		'''Stream an individual memory.'''
		
		total = []
		# Add the base memory
		memory = iter(memory)
		id = id or self.add_memory(next(memory))
		
		# Continue iterating, concatenating new tokens
		for m in memory:
			self.db.memory.concat('msg', m, id=id).commit()
			total.append(m)
			yield m
		
		# Only add to the chat log after the memory is complete
		self.add_message(''.join(total))
	
	def complete(self, prompt: str) -> str|Iterator[str]:
		'''AI prompt completion.'''
		
		self.dprint(f"complete({prompt!r})")
		return complete(prompt)
	
	def summarize(self, dialog: Diatree) -> str|Iterator[str]:
		'''Summarize the current chat log.'''
		
		conv = str(dialog)
		self.dprint(f"summarize({conv!r})")
		return self.complete(prompt.summarize(self.name, conv))
	
	def chat(self, user: str, msg: str) -> Iterator[str]:
		'''Respond to the user's message.'''
		
		msg = f"{prompt.name(user)} {msg}"
		
		self.dprint(f"chat({msg!r})")
		self.add_memory(msg)
		
		msg = prompt.name(self.name)
		yield msg
		id = self.add_memory(msg)
		yield from self.stream(self.complete(str(self.dialog)), id)
		
		# Summarize every so often
		if self.unsummarized >= CHAT_LINES * PRESSURE:
			self.unsummarized = 0
			yield from self.stream(
				chain(["[SUMMARY]"], self.summarize(self.dialog))
			)