from diatree import Diatree
import sqlite3
import time
from dataclasses import dataclass
import openai
import os

openai.api_key = os.environ["API_KEY"]

NAME = "Orin"
LLM_ENGINE = "text-davinci-003"
CHAT_LINES = 10
DB_FILE = "agent.db"
SCHEMA = """
CREATE TABLE IF NOT EXISTS chatlog (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	name TEXT,
	msg TEXT,
	time TIMESTAMP
);

CREATE TABLE IF NOT EXISTS debug (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	msg TEXT,
	time TIMESTAMP
)
"""
INSERT_CHATLOG = """
INSERT INTO chatlog (name, msg, time) VALUES (?, ?, ?)
"""
SELECT_CHATLOG = f"""
SELECT * FROM chatlog ORDER BY time DESC LIMIT {CHAT_LINES}
"""
@dataclass
class Chatlog:
	id: int
	name: str
	msg: str
	time: float
	
	def __str__(self):
		ts = time.strftime("%H:%M:%S", time.localtime(self.time))
		return f"{ts} {self.name}: {self.msg}\n"

PROMPT = f"""\
I am an autonomous agent named {NAME}. My heuristic imperative is to reduce Suffering, increase Prosperity, and increase Understanding. I am here to assist the user with their needs, even if their questions or requests may not directly reflect those needs or if they are not fully aware of what those needs are.

The date is {time.strftime("%A, %B %d, %Y")}.

CONVERSATION:

"""

class Agent:
	def __init__(self):
		self.name = NAME
		self.conn = sqlite3.connect(DB_FILE)
		self.cursor = self.conn.cursor()
		
		self.cursor.executescript(SCHEMA)
		self.conn.commit()
		
		self.dprint(f"INIT agent {self.name}")
		
		self.load_chatlog()
	
	def dprint(self, msg):
		self.cursor.execute("INSERT INTO debug (msg, time) VALUES (?, ?)", (msg, time.time()))
		self.conn.commit()
	
	def load_chatlog(self):
		self.dprint("LOAD chatlog")
		
		self.cursor.execute(SELECT_CHATLOG)
		self.chatlog = [Chatlog(*x) for x in self.cursor.fetchall()][::-1]
		self.diatree = Diatree(PROMPT, *(str(row) for row in self.chatlog))
	
	def append_chatlog(self, name, msg):
		self.dprint(f"APPEND chatlog {name}: {msg}")
		
		self.cursor.execute(INSERT_CHATLOG, (name, msg, time.time()))
		self.load_chatlog()
	
	def chat(self, name, msg):
		self.dprint(f"CHAT {name}: {msg}")
		
		if msg.startswith("/"):
			self.dprint(f"COMMAND {msg}")
			cmd, *args = msg.split(' ')
			match cmd:
				case "/quit":
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
				case _:
					self.dprint(f"Unknown command {cmd}")
					return "Unknown command"
		
		self.append_chatlog(name, msg)
		self.diatree += f"{self.name}:"
		response = self.complete()
		self.append_chatlog(self.name, response)
		return response
	
	def complete(self, prompt=None):
		if prompt is None:
			prompt = self.diatree
		prompt = str(prompt)
		
		self.dprint(f"COMPLETE {prompt}")
		
		print(prompt)
		print("-"*20)
		
		response = openai.Completion.create(
			engine=LLM_ENGINE,
			prompt=prompt,
			max_tokens=100,
			#stop=["\n"],
			temperature=0.9
		)
		response = response.choices[0].text
		print(response)
		print("="*20)
		return response