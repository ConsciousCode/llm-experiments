import numpy as np
import spacy
from dataclasses import dataclass, field

@dataclass
class SpacyTags:
	'''Lazily unpacked spacy tags.'''
	
	token: str
	spacy: spacy.tokens.Token
	
	def __iter__(self):
		token = self.spacy
		yield from (
			f"@{self.token}",
			f".{token.text}",
			f"^{token.head.text}",
			f"/{token.lemma_}",
			token.pos_.upper(),
			token.tag_.upper(),
			token.dep_.upper()
		)

@dataclass
class Tagset:
	'''A set of tags for a token, including a scoped context.'''
	
	tags: list[str]
	context: 'list[str]|Tagset'
	
	def __len__(self):
		return len(self.context) + len(self.tags)
	
	def __iter__(self):
		yield from self.context
		yield from self.tags

@dataclass
class MultiTagset:
	'''List of tagsets with a common scoped context.'''
	
	tags: list[Tagset]
	context: list[str]|Tagset = field(default_factory=list)
	
	def __len__(self): return len(self.tags)
	def __iter__(self):
		for tags in self.tags:
			yield Tagset(tags, self.context)
	
	def mask(self, mask):
		for t, m in zip(self.tags, mask):
			if m:
				yield Tagset(t, self.context)
	
	def enter(self, tags):
		'''Enter a new context.'''
		return MultiTagset(self.tags, Tagset(tags, self.context))

class Tagger:
	'''Use NLP to tag transformer tokens.'''
	
	def __init__(self, tokenizer, nlp):
		self.tokenizer = tokenizer
		self.nlp = nlp
	
	def tag(self, text: str, context: list[str]) -> tuple[np.ndarray, MultiTagset]:
		tokens = self.tokenizer(text, return_offsets_mapping=True)
		ids = np.array(tokens.input_ids)
		doc = self.nlp(text)
		token_tags, id = [], 0
		for start, end in tokens.offset_mapping:
			tags = None
			while id < len(doc):
				token = doc[id]
				if token.idx >= end:
					break
				if token.idx >= start:
					st = SpacyTags(text[start:end], token)
					tags = st if tags is None else Tagset(st, tags)
				id += 1
			
			token_tags.append(tags)
		
		return ids, MultiTagset(token_tags, context)
