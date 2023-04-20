'''
Dialog Tree data structure - immutable trees used to organize dialogues
and their alternatives. Basically just a tuple with specific mutation operators.

What composes a dialog tree?
* Prompt (root node)
* Linked list chain of current dialogues
'''

def flatten(children):
	'''
	Flattens a list of strings and lists of strings into a generator of strings.
	'''
	for child in children:
		if isinstance(child, str):
			yield child
		else:
			yield from flatten(child)

class Diatree:
	'''
	Dialog tree data structure. Acts like a tuple[str] with mutation operators.
	'''
	
	def __init__(self, *children):
		self.children = children
	
	def __str__(self): return ''.join(self.children)
	def __repr__(self): return f'Diatree{super().__repr__()}'
	def __iter__(self): return iter(self.children)
	def __len__(self): return len(self.children)
	def __add__(self, other):
		if isinstance(other, str):
			return Diatree(*self, other)
		return Diatree(*self, *other)
	
	def __getitem__(self, x: int|slice):
		if isinstance(x, slice):
			return Diatree(*super().__getitem__(x))
		return super().__getitem__(x)
	
	def alter(self, x: int|range, *elems):
		'''
		Inserts an alternative dialogue at index x.
		'''
		if isinstance(x, range):
			y = x.stop
			x = x.start
		else:
			y = x + 1
		return Diatree(*self[:x], *elems, *self[y:])
	
	def insert(self, x: int, elem):
		'''
		Inserts a new dialogue at index x.
		'''
		if x < 0:
			xl = len(self)
			x = (x + xl) % xl + 1
		return Diatree(*self[:x], elem, *self[x:])
	
	def push(self, elem):
		'''
		Inserts a new dialogue at the end of the chain.
		'''
		return self.insert(-1, elem)
	
	def pop(self, x=-1):
		'''
		Removes a dialogue at index x.
		'''
		return self.alter(x)
	

Diatree("Hello world")