import sys
from agent import Agent
from argparse import ArgumentParser
from prompt_toolkit import prompt, PromptSession

HELP = """
Commands:
  h/help - this list
  q/quit - quit
  sql ...code - execute SQL code
  select/insert/update/delete [...code] - execute SQL code with the given verb
  prompt - print the completion prompt as the AI will see it
  state [key [value]] - get or set the agent's state dictionary
  sum/summary/summarize - force a chatlog summary
  level [level] - get or set the agent's log level (0-5 or QUIET, ERROR, WARN, DEBUG, INFO, VERBOSE)
"""

def cli_type(it):
	for token in it:
		for c in token:
			print(c, end='', flush=True)

def main():
	agent = Agent()
	
	if len(sys.argv) < 2:
		name = prompt("Enter your name: ")
	else:
		name = sys.argv[1]
	
	sess = PromptSession()
	while True:
		msg = sess.prompt(f"<{name}> ")
		if msg.startswith("/"):
			cmd, *args = msg[1:].strip().split(' ')
			match cmd:
				case "h"|"help": print(HELP)
				case "q"|"quit": return
				
				case _: cli_type(agent.command(name, cmd, args))
				
			continue
		
		cli_type(agent.chat(name, msg))
		print()

if __name__ == '__main__':
	main()