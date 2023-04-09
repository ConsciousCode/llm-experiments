#!/usr/bin/env python3

print("import model")
import argparse
from model import Orin
print("import train")
import train
import os
print("import torchinfo")
import torchinfo

default_path = "orin.pt"

ap = argparse.ArgumentParser()
sp = ap.add_subparsers(dest='action')

help_parser = sp.add_parser("help", help="Print help text")
init_parser = sp.add_parser('init', help="Initialize a new Orin model")
train_parser = sp.add_parser('train', help="Train an existing model")
test_parser = sp.add_parser('test')
gen_parser = sp.add_parser('gen', help="Generate text")
checkpoint_parser = sp.add_parser('checkpoint', help="Load from a checkpoint then save as a model")
visualize_parser = sp.add_parser("visualize", help="Visualize the attention patterns")
summary_parser = sp.add_parser("summary", help="Print a summary of the model")

init_parser.add_argument('-f', "--file", default=default_path, help="Model file to load")

train_parser.add_argument("-f", "--file", default=default_path, help="Model file to load")
train_parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs to train for')

gen_parser.add_argument("init", default="", help="Initial text")
gen_parser.add_argument('-f', "--file", default=default_path, help="Model file to load")
gen_parser.add_argument('-n', "--count", type=int, default=100, help="How many characters to generate")

checkpoint_parser.add_argument("checkpoint", default="", help="The checkpoint file")
checkpoint_parser.add_argument('-f', "--file", default=default_path, help="The model to save to")

visualize_parser.add_argument("init", default="", help="Initial text")
visualize_parser.add_argument('-f', "--file", default=default_path, help="Model file to load")
visualize_parser.add_argument('-n', "--count", type=int, default=100, help="How many characters to generate")

summary_parser.add_argument('-f', '--file', default=default_path, help="Model file to load")
summary_parser.add_argument('-v', '--verbose', action="store_true", default=False, help="More verbose summary")

def save_orin(model, file):
	if os.path.exists(file):
		sel = input("Overwrite file? (y/N) ")[:1]
		if sel.lower() != 'y':
			print("Aborting.")
			exit(1)
		
	return model.save(file)

def main():
	args = ap.parse_args()
	
	file = default_path if args.file == '.' else args.file
	
	match args.action:
		case "help":
			ap.print_help()
		
		case "init":
			save_orin(Orin.init(), args.file)
		
		case "train":
			train.main(file, args.epochs)
		
		case "gen":
			try:
				says = Orin.load(file).generate(args.init, args.count)
				print(says.decode('utf-8'))
			except UnicodeDecodeError:
				print(says)
			
			except FileNotFoundError:
				print(f"Cannot access {file}: No such file")
		
		case "checkpoint":
			Orin.load(args.checkpoint).save(file)
		
		case "summary":
			m = Orin.load(file)
			if args.verbose:
				print(m)
			else:
				torchinfo.summary(m)

if __name__ == "__main__":
	main()