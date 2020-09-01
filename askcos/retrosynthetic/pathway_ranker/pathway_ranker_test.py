import json
import os
import unittest

import torch

from askcos.retrosynthetic.pathway_ranker.pathway_ranker import PathwayRanker


class TestPathwayRanker(unittest.TestCase):
    """Contains functional tests for the PathwayRanker class."""

    @classmethod
    def setUpClass(cls):
        with open(os.path.join(os.path.dirname(__file__), 'test_data', 'test_trees.json'), 'r') as f:
            cls.trees = json.load(f)

    def test_preprocess(self):
        """Test the preprocess method."""
        ranker = PathwayRanker()
        original_indices, batch = ranker.preprocess(self.trees)

        self.assertEqual(original_indices, [2, 3, 4])

        self.assertIn('pfp', batch)
        self.assertIsInstance(batch['pfp'], torch.Tensor)
        self.assertIn('rxnfp', batch)
        self.assertIsInstance(batch['rxnfp'], torch.Tensor)
        self.assertIn('node_order', batch)
        self.assertIsInstance(batch['node_order'], torch.Tensor)
        self.assertIn('adjacency_list', batch)
        self.assertIsInstance(batch['adjacency_list'], torch.Tensor)
        self.assertIn('edge_order', batch)
        self.assertIsInstance(batch['edge_order'], torch.Tensor)
        self.assertEqual(batch['num_nodes'], [2, 5, 4])
        self.assertEqual(batch['num_trees'], [1, 1, 1])
        self.assertEqual(batch['batch_size'], 3)

    def test_scorer(self):
        """Test the scorer method."""
        ranker = PathwayRanker()
        ranker.load()

        output = ranker.scorer(self.trees, clustering=True)

        self.assertIn('scores', output)
        self.assertEqual(len(output['scores']), 3)
        self.assertIn('encoded_trees', output)
        self.assertEqual(len(output['encoded_trees']), 3)
        self.assertEqual(len(output['encoded_trees'][0]), 512)
        self.assertIn('original_indices', output)
        self.assertEqual(output['original_indices'], [2, 3, 4])
        self.assertIn('clusters', output)
        self.assertEqual(output['clusters'], [0, 1, 2])


if __name__ == '__main__':
    res = unittest.main(verbosity=3, exit=False)
