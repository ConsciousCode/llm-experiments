import unittest
import numpy as np
from client import GRPCModel

class TestClient(unittest.TestCase):
	def test_tokenize(self):
		text = "This is a test sentence."
		tokens = GRPCModel().tokenize(text)
		self.assertIsInstance(tokens, list)
		self.assertIsInstance(tokens[0], int)
	
	def test_decode(self):
		text = "This is a test sentence."
		tokens = GRPCModel().tokenize(text)
		detokenized = GRPCModel().decode(tokens)
		self.assertEqual(text, detokenized)
	
	def assertProcess(self, result):
		self.assertIsInstance(result.logits, np.ndarray)
		
		self.assertIsInstance(result.hidden, list)
		self.assertIsInstance(result.hidden[0], np.ndarray)
		
		self.assertIsInstance(result.attention, list)
		self.assertIsInstance(result.attention[0], np.ndarray)
		
		# Hidden includes the initial embeddings
		self.assertEqual(len(result.hidden) - 1, len(result.attention))
	
	def test_forward_with_text(self):
		text = "This is a test sentence."
		result = GRPCModel().forward(text, True, True)
		self.assertProcess(result)

	def test_forward_with_tokens(self):
		tokens = GRPCModel().tokenize("This is a test sentence.")
		result = GRPCModel().forward(tokens, True, True)
		self.assertProcess(result)

	def test_complete(self):
		text = "This is a test sentence."
		completion = GRPCModel().complete(text, stream=False)
		self.assertIsInstance(completion, str)
		self.assertFalse(completion.startswith(text))

	def test_embed(self):
		text = "This is a test sentence."
		v = GRPCModel().embed(text)
		self.assertIsInstance(v, np.ndarray)

if __name__ == '__main__':
	unittest.main()
