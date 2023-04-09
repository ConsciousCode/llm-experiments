import unittest
import numpy as np
import torch

import sys
sys.path.insert(0, '..')

from orin.knn import KNN, KNNMemory

class TestKNN(unittest.TestCase):

    def setUp(self):
        self.dim = 128
        self.size = 100
        self.knn = KNN(dim=self.dim, size=self.size)

    def test_train(self):
        x = np.random.rand(10, self.dim).astype(np.float32)
        self.knn.train(x)
        self.assertTrue(self.knn.is_trained)

    def test_reset(self):
        x = np.random.rand(10, self.dim).astype(np.float32)
        self.knn.train(x)
        self.knn.reset()
        self.assertFalse(self.knn.is_trained)
        self.assertEqual(self.knn.ids.size, 0)

    def test_add(self):
        x = np.random.rand(10, self.dim).astype(np.float32)
        ids = np.arange(10).astype(np.int32)
        self.knn.add(x, ids)
        self.assertEqual(self.knn.ids.size, 10)

    def test_search(self):
        x = np.random.rand(10, self.dim).astype(np.float32)
        ids = np.arange(10).astype(np.int32)
        self.knn.add(x, ids)

        queries = np.random.rand(5, self.dim).astype(np.float32)
        topk = 3
        result = self.knn.search(queries, topk)
        self.assertEqual(result.shape, (5, topk))

    def test_trim_to_max_entries(self):
        x = np.random.rand(110, self.dim).astype(np.float32)
        ids = np.arange(110).astype(np.int32)
        self.knn.add(x, ids)
        self.knn.trim_to_max_entries()
        self.assertEqual(self.knn.ids.size, self.size)

class TestKNNMemory(unittest.TestCase):

    def setUp(self):
        self.dim = 8
        self.max_mems = 32
        self.num_indices = 4
        self.knn_memory = KNNMemory(dim=self.dim, max_mems=self.max_mems, num_indices=self.num_indices)

    def test_add_and_search(self):
        batch_size = 3
        mems = torch.randn(batch_size, self.max_mems, 2, self.dim)

        # Set scoped_indices and add mems
        self.knn_memory.set_scoped_indices([0, 1, 2])
        self.knn_memory.add(mems)

        # Create queries
        queries = torch.randn(batch_size, self.dim)

        # Search top-k nearest neighbors
        topk = 5
        key_values, masks = self.knn_memory.search(queries, topk)

        # Check the shapes of key_values and masks
        self.assertEqual(key_values.shape, (batch_size, topk, 2, self.dim))
        self.assertEqual(masks.shape, (batch_size, topk))

    def test_scope_indices(self):
        batch_size = 3
        mems = torch.randn(batch_size, self.max_mems, 2, self.dim)

        # Set scoped_indices and add mems
        self.knn_memory.set_scoped_indices([0, 1, 2])
        self.knn_memory.add(mems)

        # Test scope_indices context manager
        with self.knn_memory.scope_indices([1, 2]):
            self.assertEqual(self.knn_memory.scoped_indices, [1, 2])

        # scoped_indices should revert back to the previous value
        self.assertEqual(self.knn_memory.scoped_indices, [0, 1, 2])

    def test_clear(self):
        batch_size = 3
        mems = torch.randn(batch_size, self.max_mems, 2, self.dim)

        # Set scoped_indices and add mems
        self.knn_memory.set_scoped_indices([0, 1, 2])
        self.knn_memory.add(mems)

        # Clear scoped_indices 1 and 2
        self.knn_memory.clear([1, 2])

        # Create queries
        queries = torch.randn(batch_size, self.dim)

        # Search top-k nearest neighbors
        topk = 5
        key_values, masks = self.knn_memory.search(queries, topk)

        # The second and third batch indices should have no results (all masks should be False)
        self.assertTrue(torch.all(~masks[1]))
        self.assertTrue(torch.all(~masks[2]))


if __name__ == "__main__":
    unittest.main()
