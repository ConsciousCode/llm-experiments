import numpy as np
import spacy

class TagContext:
	def __init__(self, tag, context):
		self.tag = tag
		self.context = context

class Tagger:
	'''
	Use NLP to tag transformer tokens
	'''
	def __init__(self, tokenizer, nlp):
		self.tokenizer = tokenizer
		self.nlp = nlp
	
	def tag(self, input_text: str, context) -> TagContext:
		tokens = self.tokenizer(input_text, return_offsets_mapping=True)
		ids = np.array(tokens.input_ids)
		doc = self.nlp(input_text)
		token_tags, id = [], 0
		for start, end in tokens.offset_mapping:
			tags = []
			while id < len(doc):
				token = doc[id]
				if token.idx >= end:
					break
				if token.idx >= start:
					tags.append(token)
				id += 1
			
			token_tags.append(tags)
		
		return TagContext(ids, token_tags, context)
