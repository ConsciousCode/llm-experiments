import sys
from agent import Agent
from argparse import ArgumentParser
from prompt_toolkit import prompt, PromptSession

HELP = """
Commands:
  h/help
  q/quit
  debug
  sql
  dia/diatree
"""

def cli_type(it):
	for token in it:
		print(token, end='', flush=True)

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
			agent.dprint(f"COMMAND {msg}")
			cmd, *args = msg[1:].split(' ')
			match cmd:
				case "h"|"help": print(HELP)
				case "q"|"quit": return
				
				case _: cli_type(agent.command(name, cmd, args))
				
			continue
		
		cli_type(agent.chat(name, msg))
		print()

if __name__ == '__main__':
	main()