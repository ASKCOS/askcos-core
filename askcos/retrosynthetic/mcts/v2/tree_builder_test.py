import unittest

import networkx as nx
import numpy as np
import rdkit.Chem as Chem

from askcos.retrosynthetic.mcts.v2.tree_builder import MCTS


class DummyHistorian:
    def lookup_smiles(self, *args, **kwargs):
        return {'as_reactant': 0, 'as_product': 0}


class TestMCTS(unittest.TestCase):
    """Contains functional tests for the MCTS class."""

    @classmethod
    def setUpClass(cls):
        """This method is run once before all tests in this class."""
        cls.mcts = MCTS(chemhistorian=DummyHistorian())
        cls.target = 'CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21'

    def tearDown(self):
        """This method is run after every test in this class."""
        self.mcts.clear()  # Clear tree
        self.mcts.set_options()  # Reset options to default

    def test_is_terminal_1(self):
        """Test that the is_terminal method works."""
        options = {
            'max_ppg': None,
        }
        self.mcts.set_options(**options)
        self.assertFalse(self.mcts.is_terminal('C'))

    def test_is_terminal_2(self):
        """Test that the is_terminal method works."""
        options = {
            'max_ppg': 10,
        }
        self.mcts.set_options(**options)
        self.assertTrue(self.mcts.is_terminal('C', 1))
        self.assertFalse(self.mcts.is_terminal('C', 12))
        self.assertFalse(self.mcts.is_terminal('C'))

    def test_is_terminal_3(self):
        """Test that the is_terminal method works."""
        options = {
            'max_scscore': 2,
        }
        self.mcts.set_options(**options)
        self.assertTrue(self.mcts.is_terminal('Brc1ccccc1'))
        self.assertFalse(self.mcts.is_terminal('CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21'))

    def test_is_terminal_4(self):
        """Test that the is_terminal method works."""
        options = {
            'max_elements': {'C': 2, 'N': 2},
        }
        self.mcts.set_options(**options)
        self.assertTrue(self.mcts.is_terminal('C'))
        self.assertTrue(self.mcts.is_terminal('NCCN'))
        self.assertFalse(self.mcts.is_terminal('CCCN'))
        self.assertFalse(self.mcts.is_terminal('NC(N)CN'))
        self.assertFalse(self.mcts.is_terminal('NCC(N)CN'))

    def test_is_terminal_5(self):
        """Test that the is_terminal method works."""
        options = {
            'min_history': {'as_reactant': 10, 'as_product': 10},
        }
        self.mcts.set_options(**options)
        self.assertTrue(self.mcts.is_terminal('C', hist={'as_reactant': 12, 'as_product': 12}))
        self.assertTrue(self.mcts.is_terminal('C', hist={'as_reactant': 12, 'as_product': 0}))
        self.assertTrue(self.mcts.is_terminal('C', hist={'as_reactant': 0, 'as_product': 12}))
        self.assertFalse(self.mcts.is_terminal('C', hist={'as_reactant': 0, 'as_product': 0}))
        self.assertFalse(self.mcts.is_terminal('C'))

    def test_is_terminal_6(self):
        """Test that the is_terminal method works."""
        options = {
            'max_ppg': 10,
            'min_history': {'as_reactant': 10, 'as_product': 10},
            'termination_logic': {'max_ppg': 'and', 'min_history': 'and'}
        }
        self.mcts.set_options(**options)
        self.assertTrue(self.mcts.is_terminal('C', ppg=1, hist={'as_reactant': 12, 'as_product': 12}))
        self.assertFalse(self.mcts.is_terminal('C', ppg=12, hist={'as_reactant': 12, 'as_product': 12}))
        self.assertFalse(self.mcts.is_terminal('C', ppg=1, hist={'as_reactant': 0, 'as_product': 0}))
        self.assertFalse(self.mcts.is_terminal('C', ppg=12, hist={'as_reactant': 0, 'as_product': 0}))

    def test_is_terminal_7(self):
        """Test that the is_terminal method works."""
        options = {
            'max_ppg': 10,
            'min_history': {'as_reactant': 10, 'as_product': 10},
            'termination_logic': {'max_ppg': 'and', 'min_history': 'or'}
        }
        self.mcts.set_options(**options)
        self.assertTrue(self.mcts.is_terminal('C', ppg=1, hist={'as_reactant': 12, 'as_product': 12}))
        self.assertTrue(self.mcts.is_terminal('C', ppg=12, hist={'as_reactant': 12, 'as_product': 12}))
        self.assertTrue(self.mcts.is_terminal('C', ppg=1, hist={'as_reactant': 0, 'as_product': 0}))
        self.assertFalse(self.mcts.is_terminal('C', ppg=12, hist={'as_reactant': 0, 'as_product': 0}))

    def test_is_terminal_8(self):
        """Test that the is_terminal method works."""
        options = {
            'max_ppg': 10,
            'max_elements': {'C': 2, 'N': 2},
            'min_history': {'as_reactant': 10, 'as_product': 10},
            'termination_logic': {'max_ppg': 'and', 'max_elements': 'and', 'min_history': 'or'}
        }
        self.mcts.set_options(**options)
        self.assertTrue(self.mcts.is_terminal('C', ppg=1, hist={'as_reactant': 12, 'as_product': 12}))
        self.assertTrue(self.mcts.is_terminal('C', ppg=1, hist={'as_reactant': 0, 'as_product': 0}))
        self.assertTrue(self.mcts.is_terminal('NCC(N)CN', ppg=1, hist={'as_reactant': 12, 'as_product': 12}))
        self.assertTrue(self.mcts.is_terminal('C', ppg=12, hist={'as_reactant': 12, 'as_product': 12}))
        self.assertTrue(self.mcts.is_terminal('NCC(N)CN', ppg=12, hist={'as_reactant': 12, 'as_product': 12}))
        self.assertFalse(self.mcts.is_terminal('NCC(N)CN', ppg=12, hist={'as_reactant': 0, 'as_product': 0}))

    def test_initialize(self):
        """Test that the _initialize method works."""
        self.mcts._initialize(self.target)
        self.assertEqual(self.mcts.target, self.target)
        self.assertIn(self.target, self.mcts.chemicals)
        self.assertIn(self.target, self.mcts.tree)

        node_data = self.mcts.tree.nodes[self.target]
        self.assertIsInstance(node_data, dict)
        self.assertFalse(node_data['terminal'])
        self.assertFalse(node_data['done'])

    def test_create_chemical_node(self):
        """Test that the create_chemical_node method works."""
        self.mcts.create_chemical_node(self.target)
        self.assertIn(self.target, self.mcts.chemicals)
        self.assertIn(self.target, self.mcts.tree)

        node_data = self.mcts.tree.nodes[self.target]
        self.assertIsInstance(node_data, dict)
        self.assertEqual(node_data['est_value'], 1)
        self.assertEqual(node_data['explored'], [])
        self.assertIsNone(node_data['min_depth'])
        self.assertEqual(node_data['purchase_price'], 30)
        self.assertIsInstance(node_data['templates'], dict)
        self.assertEqual(len(node_data['templates']), 78)
        self.assertTrue(node_data['terminal'])
        self.assertEqual(node_data['type'], 'chemical')
        self.assertEqual(node_data['visit_count'], 1)
        self.assertTrue(node_data['done'])

    def test_create_reaction_node(self):
        """Test that the create_reaction_node method works."""
        rxn = 'BrBr.c1ccccc1>>Brc1ccccc1'
        self.mcts.create_reaction_node(rxn, 42, 0.5, 0.9)
        self.assertIn(rxn, self.mcts.reactions)
        self.assertIn(rxn, self.mcts.tree)

        node_data = self.mcts.tree.nodes[rxn]
        self.assertIsInstance(node_data, dict)
        self.assertEqual(node_data['est_value'], 0)
        self.assertEqual(node_data['ff_score'], 0.9)
        self.assertEqual(node_data['template_score'], 0.5)
        self.assertEqual(node_data['templates'], [42])
        self.assertEqual(node_data['type'], 'reaction')
        self.assertEqual(node_data['visit_count'], 1)

    def test_get_precursors(self):
        """Test that the _get_precursors method works."""
        chem = self.target
        template = 5527
        precursors = self.mcts._get_precursors(chem, template)
        expected_precursors = [['COS(=O)(=O)OC', 'O=C1CN=C(c2ccccc2)c2cc(Cl)ccc2N1']]
        self.assertEqual(precursors, expected_precursors)

    def test_process_precursors(self):
        """Test that the _process_precursors method works."""
        chem = self.target
        self.mcts.create_chemical_node(chem)
        template = 5527
        c1 = 'COS(=O)(=O)OC'
        c2 = 'O=C1CN=C(c2ccccc2)c2cc(Cl)ccc2N1'
        precursors = [[c1, c2]]
        path = [chem]

        self.mcts._process_precursors(chem, template, precursors, path)

        self.assertEqual(len(self.mcts.tree), 4)

        # Check chemical nodes
        self.assertEqual(len(self.mcts.chemicals), 3)

        self.assertIn(c1, self.mcts.chemicals)
        node_data = self.mcts.tree.nodes[c1]
        self.assertIsInstance(node_data['templates'], dict)
        self.assertEqual(node_data['purchase_price'], 12)

        self.assertIn(c2, self.mcts.chemicals)
        node_data = self.mcts.tree.nodes[c2]
        self.assertIsInstance(node_data['templates'], dict)
        self.assertEqual(node_data['purchase_price'], 26)

        # Check reaction nodes
        self.assertEqual(len(self.mcts.reactions), 1)
        rxn = '.'.join(precursors[0]) + '>>' + self.target
        self.assertIn(rxn, self.mcts.reactions)
        node_data = self.mcts.tree.nodes[rxn]

        self.assertAlmostEqual(node_data['ff_score'], 0.99978, places=3)
        self.assertAlmostEqual(node_data['template_score'], 0.13412, places=3)
        self.assertEqual(node_data['templates'], [template])

        # Check connectivity
        self.assertEqual(len(self.mcts.tree.edges), 3)
        self.assertTrue(self.mcts.tree.has_edge(chem, rxn))
        self.assertTrue(self.mcts.tree.has_edge(rxn, c1))
        self.assertTrue(self.mcts.tree.has_edge(rxn, c2))

    def test_rollout(self):
        """Test that the _rollout method works."""
        self.mcts._initialize(self.target)

        self.mcts._rollout()
        self.assertEqual(self.mcts.tree.nodes[self.target]['visit_count'], 2)
        self.assertEqual(len(self.mcts.chemicals), 2)
        self.assertEqual(len(self.mcts.reactions), 1)

        self.mcts._rollout()
        self.assertEqual(self.mcts.tree.nodes[self.target]['visit_count'], 3)
        self.assertEqual(len(self.mcts.chemicals), 4)
        self.assertEqual(len(self.mcts.reactions), 2)

    def test_build_tree(self):
        """Test that the build_tree method works."""
        self.mcts.build_tree(
            self.target,
            max_chemicals=10,  # Expand until 10 chemicals have been explored
        )

        self.assertEqual(len(self.mcts.chemicals), 10)
        self.assertEqual(len(self.mcts.reactions), 7)


if __name__ == '__main__':
    res = unittest.main(verbosity=3, exit=False)
