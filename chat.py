import sys
from agent import Agent
from argparse import ArgumentParser

def main():
	agent = Agent()
	
	if len(sys.argv) < 2:
		name = input("Enter your name: ")
	else:
		name = sys.argv[1]
	
	while True:
		msg = input(f"{name}: ")
		print(f"{agent.name}: {agent.chat(name, msg)}")

if __name__ == '__main__':
	main()