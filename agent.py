from diatree import Diatree
import sqlite3
import time
from dataclasses import dataclass
import openai
import os

openai.api_key = os.environ["API_KEY"]

DEBUG = True
NAME = "Orin"
LLM_ENGINE = "text-davinci-003"
CHAT_LINES = 25
DB_FILE = "agent.db"

HELP = """
Commands:
  h/help
  q/quit
  debug
  sql
  dia/diatree
"""

SCHEMA = """
CREATE TABLE IF NOT EXISTS memory (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	msg TEXT,
	time TIMESTAMP
);

CREATE TABLE IF NOT EXISTS debug (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	msg TEXT,
	time TIMESTAMP
)
"""
INSERT_MEMORY = """
INSERT INTO memory (msg, time) VALUES (?, ?)
"""
SELECT_MEMORY = f"""
SELECT * FROM memory ORDER BY time DESC LIMIT {CHAT_LINES}
"""
@dataclass
class Chatlog:
	id: int
	msg: str
	time: float
	
	def __str__(self):
		ts = time.strftime("%H:%M:%S", time.localtime(self.time))
		return f"{ts} {self.msg}\n"

MEMORY_COUNT = "SELECT COUNT(*) FROM memory"
INSERT_DEBUG = "INSERT INTO debug (msg, time) VALUES (?, ?)"

PROMPT = f"""\
I am an autonomous agent named {NAME}. My heuristic imperative is to reduce Suffering, increase Prosperity, and increase Understanding. I am here to assist the user with their needs, even if their questions or requests may not directly reflect those needs or if they are not fully aware of what those needs are.

The date is {time.strftime("%A, %B %d, %Y")}.

CONVERSATION:

"""

SUMMARIZE = """\
Given the following conversation, provide a salient summary:

{input}

SUMMARY:
"""

class Agent:
	def __init__(self):
		self.name = NAME
		self.conn = sqlite3.connect(DB_FILE)
		self.cursor = self.conn.cursor()
		self.debug = DEBUG
		
		self.cursor.executescript(SCHEMA)
		self.conn.commit()
		
		self.unsummarized = 0
		
		self.dprint(f"INIT agent {self.name}")
		
		self.add_memory("-- SYSTEM: Program reloaded --")
		self.reload()
	
	def dprint(self, msg):
		self.cursor.execute(INSERT_DEBUG, (msg, time.time()))
		self.conn.commit()
	
	def reload(self):
		self.dprint("RELOAD")
		
		self.cursor.execute(SELECT_MEMORY)
		self.memory = [Chatlog(*x) for x in self.cursor.fetchall()][::-1]
		self.diatree = Diatree(PROMPT, *(str(row) for row in self.memory))
	
	def add_memory(self, msg):
		self.dprint(f"ADD memory {msg}")
		
		self.unsummarized += 1
		self.cursor.execute(INSERT_MEMORY, (msg, time.time()))
		self.reload()
	
	def summarize(self):
		conv = str(self.diatree[1:])
		
		self.dprint(f"SUMMARIZE {conv}")
		
		response = openai.Completion.create(
			engine=LLM_ENGINE,
			prompt=SUMMARIZE.format(input=conv),
			max_tokens=250,
			temperature=0.2
		)
		return response.choices[0].text
	
	def chat(self, name, msg):
		if msg.startswith("/"):
			self.dprint(f"COMMAND {msg}")
			cmd, *args = msg.split(' ')
			match cmd:
				case "/h"|"/help":
					return HELP
				case "/q"|"/quit":
					exit()
				case "/sql":
					try:
						self.cursor.execute(' '.join(args))
						self.conn.commit()
						return str(self.cursor.fetchall())
					except sqlite3.OperationalError as e:
						self.dprint(f"SQL ERROR:\n{e}")
						return str(e)
				case "/dia"|"/diatree":
					return str(self.diatree)
				case "/debug":
					self.debug = not self.debug
				case _:
					self.dprint(f"Unknown command {cmd}")
					return "Unknown command"
			
			return "FALLTHROUGH"
		
		msg = f"<{name}> {msg}"
		self.dprint(f"CHAT {msg}")
		self.add_memory(msg)
		self.diatree += f"{time.strftime('%H:%M:%S')} <{self.name}>"
		response = f"<{self.name}>{self.complete()}"
		self.add_memory(response)
		
		print("COUNT", self.unsummarized)
		if self.unsummarized >= CHAT_LINES // 2:
			self.unsummarized = 0
			summary = f"[SUMMARY] {self.summarize()}"
			self.add_memory(summary)
			if self.debug:
				response += f"\n{summary}"
		
		self.reload()
		
		return response
	
	def complete(self, prompt=None):
		if prompt is None:
			prompt = self.diatree
		prompt = str(prompt)
		
		self.dprint(f"COMPLETE {prompt}")
		
		response = openai.Completion.create(
			engine=LLM_ENGINE,
			prompt=prompt,
			max_tokens=100,
			#stop=["\n"],
			temperature=0.9
		)
		return response.choices[0].text