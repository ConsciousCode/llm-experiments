import unittest
import numpy as np
from client import tokenize, tokendecode, process, complete, embed

class TestClient(unittest.TestCase):
	def test_tokenize(self):
		text = "This is a test sentence."
		tokens = tokenize(text)
		self.assertIsInstance(tokens, list)
		self.assertIsInstance(tokens[0], int)
	
	def test_token_decode(self):
		text = "This is a test sentence."
		tokens = tokenize(text)
		detokenized = tokendecode(tokens)
		self.assertEqual(text, detokenized)
	
	def assertProcess(self, result):
		self.assertIsInstance(result.logits, np.ndarray)
		
		self.assertIsInstance(result.hidden, list)
		self.assertIsInstance(result.hidden[0], np.ndarray)
		
		self.assertIsInstance(result.attention, list)
		self.assertIsInstance(result.attention[0], np.ndarray)
		
		# Hidden includes the initial embeddings
		self.assertEqual(len(result.hidden) - 1, len(result.attention))
	
	def test_process_with_text(self):
		text = "This is a test sentence."
		result = process(text, True, True)
		self.assertProcess(result)

	def test_process_with_tokens(self):
		tokens = tokenize("This is a test sentence.")
		result = process(tokens, True, True)
		self.assertProcess(result)

	def test_complete(self):
		text = "This is a test sentence."
		completion = complete(text, stream=False)
		self.assertIsInstance(completion, str)
		self.assertFalse(completion.startswith(text))

	def test_embed(self):
		text = "This is a test sentence."
		v = embed(text)
		self.assertIsInstance(v, np.ndarray)

if __name__ == '__main__':
	unittest.main()
