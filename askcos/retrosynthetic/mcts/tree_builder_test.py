import unittest

import numpy as np
import rdkit.Chem as Chem

from askcos.retrosynthetic.mcts.tree_builder import MCTS


class TestMCTSTreeBuilder(unittest.TestCase):
    """Contains functional tests for the MCTS class."""

    @classmethod
    def setUpClass(cls):
        cls.nproc = 2
        cls.mcts = MCTS(nproc=cls.nproc)

    def assertDictAlmostEqual(self, first, second, rtol=1e-7, atol=0, msg=None, ignore_keys=None):
        """
        Recursive dictionary comparison, wth the following properties:
        - dict keys must be exactly equal, order is ignored
        - numerical dict values are compared using np.allclose
        - non-numerical dict values must be exactly equal
        - lists are assumed to be homogeneous, and the contents are compared
        """
        self.assertEqual(set(first.keys()), set(second.keys()))

        for key, value in first.items():
            if ignore_keys and key in ignore_keys:
                continue
            if isinstance(value, dict):
                self.assertDictAlmostEqual(value, second[key], rtol=rtol, atol=atol, msg=msg, ignore_keys=ignore_keys)
            elif isinstance(value, (str, bool)):
                self.assertEqual(value, second[key], msg=msg)
            elif isinstance(value, (int, float, np.ndarray, np.generic)):
                np.testing.assert_allclose(value, second[key], rtol=rtol, atol=atol, err_msg=msg)
            elif isinstance(value, list):
                self.assertListAlmostEqual(value, second[key], rtol=rtol, atol=atol, msg=msg, ignore_keys=ignore_keys)
            else:
                raise NotImplementedError(type(value))

    def assertListAlmostEqual(self, first, second, rtol=1e-7, atol=0, msg=None, ignore_keys=None):
        """
        Recursive list comparison, with the following properties:
        - contents are assumed to be of homogeneous type
        - numerical values are compared using np.allclose
        - non-numerical values must be exactly equal
        """
        self.assertEqual(len(first), len(second))

        if len(first) == 0:
            # Empty list
            return

        if isinstance(first[0], dict):
            for a, b in zip(first, second):
                self.assertDictAlmostEqual(a, b, rtol=rtol, atol=atol, msg=msg, ignore_keys=ignore_keys)
        elif isinstance(first[0], (str, bool)):
            self.assertEqual(first, second, msg=msg)
        elif isinstance(first[0], (int, float, np.ndarray, np.generic)):
            np.testing.assert_allclose(first, second, rtol=rtol, atol=atol, err_msg=msg)
        elif isinstance(first[0], list):
            for a, b in zip(first, second):
                self.assertListAlmostEqual(a, b, rtol=rtol, atol=atol, msg=msg, ignore_keys=ignore_keys)
        else:
            raise NotImplementedError(type(first[0]))

    def test_diphenhydramine(self):
        """Test that MCTS works for diphenhydramine."""
        self.assertTrue(self.mcts)

        smiles = 'CN(C)CCOC(c1ccccc1)c1ccccc1'
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), True)
        status_result, paths_result = self.mcts.get_buyable_paths(
            smiles,
            nproc=self.nproc,
            expansion_time=30,
            max_cum_template_prob=0.995,
            template_count=100,
            soft_reset=True,  # do not kill workers since there are more tests
            soft_stop=True,  # do not kill workers since there are more tests
        )

        # Check that the number of chemical and reactions is in a reasonable range
        # Use very conservative lower bound to account for non-deterministic behavior
        self.assertGreater(status_result[0], 40)
        self.assertGreater(status_result[1], 200)

        # Check that the top path result matches the expected one
        output_path_1 = paths_result[0]
        expected_path_1 = {
            'as_reactant': 59,
            'children': [
                {'id': 87,
                 'tforms': ['5e1f4b6e6348832850997243'],
                 'template_score': 0.07315371,
                 'necessary_reagent': '',
                 'children': [
                     {'as_reactant': 2438,
                      'children': [],
                      'ppg': 1.0,
                      'is_chemical': True,
                      'id': 79,
                      'as_product': 102,
                      'smiles': 'BrC(c1ccccc1)c1ccccc1'},
                     {'as_reactant': 3643,
                      'children': [],
                      'ppg': 1.0,
                      'is_chemical': True,
                      'id': 84,
                      'as_product': 383,
                      'smiles': 'CN(C)CCO'}
                 ],
                 'num_examples': 185,
                 'plausibility': 0.9988686442375183,
                 'is_reaction': True,
                 'smiles': 'BrC(c1ccccc1)c1ccccc1.CN(C)CCO>>CN(C)CCOC(c1ccccc1)c1ccccc1'
                 }
            ],
            'ppg': 1.0,
            'is_chemical': True,
            'id': 4,
            'as_product': 44,
            'smiles': 'CN(C)CCOC(c1ccccc1)c1ccccc1'
        }

        # The 'id' field is non-deterministic, so ignore it
        self.assertDictAlmostEqual(output_path_1, expected_path_1, rtol=1e-3, ignore_keys=['id'],
                                   msg='Obtained different output: {}'.format(output_path_1))

    def test_scopolamine(self):
        """Test that MCTS works for scopolamine."""
        self.assertTrue(self.mcts)

        smiles = 'Cc1ncc([N+](=O)[O-])n1CC(C)O'
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), True)
        status_result, paths_result = self.mcts.get_buyable_paths(
            smiles,
            nproc=self.nproc,
            expansion_time=30,
            max_cum_template_prob=0.995,
            template_count=100,
            soft_reset=True,  # do not kill workers since there are more tests
            soft_stop=True,  # do not kill workers since there are more tests
        )

        # Check that the number of chemical and reactions is in a reasonable range
        # Use very conservative lower bound to account for non-deterministic behavior
        self.assertGreater(status_result[0], 60)
        self.assertGreater(status_result[1], 200)

        # Check that the top path result matches the expected one
        output_path_1 = paths_result[0]
        expected_path_1 = {
            'id': 15,
            'ppg': 14.0,
            'children': [
                {'template_score': 0.02059098,
                 'num_examples': 349,
                 'tforms': ['5e1f4b6e63488328509980e4', '5e1f4b6f634883285099c313', '5e1f4b7063488328509af8e0', '5e1f4b6e6348832850997239'],
                 'children': [
                     {'id': 34,
                      'ppg': 1.0,
                      'children': [],
                      'as_product': 1027,
                      'as_reactant': 5560,
                      'smiles': 'CC1CO1',
                      'is_chemical': True},
                     {'id': 11,
                      'ppg': 1.0,
                      'children': [],
                      'as_product': 33,
                      'as_reactant': 169,
                      'smiles': 'Cc1ncc([N+](=O)[O-])[nH]1',
                      'is_chemical': True}
                 ],
                 'is_reaction': True,
                 'id': 35,
                 'plausibility': 0.9970303773880005,
                 'smiles': 'CC1CO1.Cc1ncc([N+](=O)[O-])[nH]1>>Cc1ncc([N+](=O)[O-])n1CC(C)O',
                 'necessary_reagent': ''}
            ],
            'as_product': 8,
            'as_reactant': 80,
            'smiles': 'Cc1ncc([N+](=O)[O-])n1CC(C)O',
            'is_chemical': True
        }

        # The 'id' field is non-deterministic, so ignore it
        self.assertDictAlmostEqual(output_path_1, expected_path_1, rtol=1e-3, ignore_keys=['id'],
                                   msg='Obtained different output: {}'.format(output_path_1))

    def test_random(self):
        """Test that MCTS works for a random compound."""
        self.assertTrue(self.mcts)

        smiles = 'CCCCCN(CCCCC)CCCC(=O)OCCC'
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), True)
        status_result, paths_result = self.mcts.get_buyable_paths(
            smiles,
            nproc=self.nproc,
            expansion_time=30,
            max_cum_template_prob=0.995,
            template_count=100,
            soft_reset=True,  # do not kill workers since there are more tests
            soft_stop=True,  # do not kill workers since there are more tests
        )

        # Check that the number of chemical and reactions is in a reasonable range
        # Use very conservative lower bound to account for non-deterministic behavior
        self.assertGreater(status_result[0], 100)
        self.assertGreater(status_result[1], 200)

        # Check that the top path result matches the expected one
        output_path_1 = paths_result[0]
        expected_path_1 = {
            'id': 12,
            'ppg': 0.0,
            'children': [
                {'template_score': 0.046091113,
                 'num_examples': 328,
                 'tforms': ['5e1f4b6e63488328509968d4'],
                 'children': [
                     {'id': 202,
                      'ppg': 0.0,
                      'children': [
                          {'template_score': 0.16923832893371582,
                           'num_examples': 4701,
                           'tforms': ['5e1f4b6e6348832850995e1d'],
                           'children': [
                               {'id': 56,
                                'ppg': 2.0,
                                'children': [],
                                'as_product': 100,
                                'as_reactant': 616,
                                'smiles': 'CCCCCNCCCCC',
                                'is_chemical': True},
                               {'id': 209,
                                'ppg': 1.0,
                                'children': [],
                                'as_product': 36,
                                'as_reactant': 651,
                                'smiles': 'N#CCCCCl',
                                'is_chemical': True}
                           ],
                           'is_reaction': True,
                           'id': 210,
                           'plausibility': 0.9925096035003662,
                           'smiles': 'CCCCCNCCCCC.N#CCCCCl>>CCCCCN(CCCC#N)CCCCC',
                           'necessary_reagent': ''}
                      ],
                      'as_product': 0,
                      'as_reactant': 0,
                      'smiles': 'CCCCCN(CCCC#N)CCCCC',
                      'is_chemical': True},
                     {'id': 10,
                      'ppg': 1.0,
                      'children': [],
                      'as_product': 2402,
                      'as_reactant': 11336,
                      'smiles': 'CCCO',
                      'is_chemical': True}
                 ],
                 'is_reaction': True,
                 'id': 203,
                 'plausibility': 0.9963662624359131,
                 'smiles': 'CCCCCN(CCCC#N)CCCCC.CCCO>>CCCCCN(CCCCC)CCCC(=O)OCCC',
                 'necessary_reagent': '[O]'}
            ],
            'as_product': 0,
            'as_reactant': 0,
            'smiles': 'CCCCCN(CCCCC)CCCC(=O)OCCC',
            'is_chemical': True
        }

        # The 'id' field is non-deterministic, so ignore it
        self.assertDictAlmostEqual(output_path_1, expected_path_1, rtol=1e-3, ignore_keys=['id'],
                                   msg='Obtained different output: {}'.format(output_path_1))

    @classmethod
    def tearDownClass(cls):
        for p in cls.mcts.workers:
            p.terminate()


if __name__ == '__main__':
    res = unittest.main(verbosity=3, exit=False)
