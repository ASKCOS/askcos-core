import json
import os
import unittest

import torch

from askcos.retrosynthetic.pathway_ranker.pathway_ranker import PathwayRanker
from askcos.retrosynthetic.pathway_ranker.utils import convert_askcos_trees


class TestPathwayRanker(unittest.TestCase):
    """Contains functional tests for the PathwayRanker class."""

    @classmethod
    def setUpClass(cls):
        with open(os.path.join(os.path.dirname(__file__), 'test_data', 'test_trees.json'), 'r') as f:
            cls.trees = json.load(f)

    def test_preprocess(self):
        """Test the preprocess method."""
        output = convert_askcos_trees(self.trees)
        original_indices, remaining_trees = zip(*((i, tree) for i, tree in enumerate(output) if tree['depth'] > 1))

        ranker = PathwayRanker()
        batch = ranker.preprocess(remaining_trees)

        self.assertEqual(original_indices, (2, 3, 4))

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
        self.assertEqual(len(output['scores']), 5)
        self.assertEqual(output['scores'][0], -1)
        self.assertEqual(output['scores'][1], -1)

        self.assertIn('encoded_trees', output)
        self.assertEqual(len(output['encoded_trees']), 5)
        self.assertEqual(len(output['encoded_trees'][0]), 0)
        self.assertEqual(len(output['encoded_trees'][1]), 0)
        self.assertEqual(len(output['encoded_trees'][2]), 512)
        self.assertEqual(len(output['encoded_trees'][3]), 512)
        self.assertEqual(len(output['encoded_trees'][4]), 512)

        self.assertIn('clusters', output)
        self.assertEqual(output['clusters'], [-1, -1, 0, 1, 2])


if __name__ == '__main__':
    res = unittest.main(verbosity=3, exit=False)
