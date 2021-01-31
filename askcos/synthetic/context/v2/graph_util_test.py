import unittest

from . import graph_util


class Test(unittest.TestCase):
    def test_01_encode_onehot(self):
        num_class = {8: 0, 53: 1, 6: 2, 7: 3, None: 4}
        self.assertEqual(graph_util.encode_onehot(8, num_class), [1, 0, 0, 0, 0])
        self.assertEqual(graph_util.encode_onehot(0, num_class), [0, 0, 0, 0, 1])
        self.assertEqual(graph_util.encode_onehot(7, num_class), [0, 0, 0, 1, 0])

    def test_encode_atom(self):
        s = {'max_atoms': 34, 'max_bonds': 35, 'max_ring_size': 6, 'max_rings': 1, 'max_bonds_per_atom': 4, 'enumerate_list_atom': {'atom_num': {0: 0, 6: 1, 7: 2, 8: 3, 53: 4, None: 5}, 'formal_charge': {0: 0, None: 1}, 'explicit_valence': {1: 0, 2: 1, 3: 2, 4: 3, None: 4}, 'implicit_valence': {0: 0, None: 1}, 'degree_conn': {1: 0, 2: 1, 3: 2, 4: 3, None: 4}, 'degree_total_conn': {1: 0, 2: 1, 3: 2, 4: 3, None: 4}, 'ring_size': {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, None: 8}}, 'bond_order': {10: 0, 20: 1, 15: 2, None: 3}}
        a = {
            "atom_idx": 8,
            "atom_map_id": 26,
            "atom_num": 6,
            "aromaticity": True,
            "formal_charge": 0,
            "explicit_valence": 4,
            "implicit_valence": 0,
            "degree_conn": 2,
            "degree_total_conn": 3,
            "in_ring": True,
            "ring_size": [
                6
                ],
            "bonds_idx": [
                7,
                8
                ]
            }
        self.assertEqual(graph_util.build_oneatom_feature(a, s).tolist(), [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_encode_bond(self):
        s = {'bond_order': {10: 0, 20: 1, 15: 2, None: 3}}
        b = {'order':20, 'is_conj':True, 'in_ring':False}
        self.assertEqual(graph_util.build_onebond_feature(b, s).tolist(), [1, 0, 0, 1, 0, 0])

    def test_encode_features(self):
        s2='[CH3:12][C:11]([CH3:13])([CH3:14])[O:10][C:8](=[O:9])[N:3]1[CH2:2][CH2:1][CH:6]([CH2:5][C:4]1=[O:7])OS(=O)(=O)C.[CH:20]1=[C:19]([N:18]=[CH:17][C:16]2=[C:15]1[CH:24]=[N:23][NH:22]2)[Cl:21]>>[CH3:12][C:11]([CH3:13])([CH3:14])[O:10][C:8](=[O:9])[N:3]1[CH2:2][CH2:1][CH:6]([CH2:5][C:4]1=[O:7])[N:22]2[C:16]3=[C:15]([CH:20]=[C:19]([N:18]=[CH:17]3)[Cl:21])[CH:24]=[N:23]2'
        s3='[CH3:1][CH2:2][CH2:3][CH2:4][CH2:5][CH2:6][CH2:7][CH2:8][CH2:9][CH2:10][CH2:11][CH2:12][CH2:13][CH2:14][CH2:15][CH2:16][CH2:17][C:18](=[O:19])O>CCCCCCCCCCCCCCCCCC(=O)O[C@H](C1=CC=CC=C1)C(=O)OC2=CC[C@]3([C@H]4CC5=CC=C(C6=C5[C@]3([C@H]2O6)CCN4C)O)O.CN(C)C=O.C1CCOC1.C(=O)(C(=O)Cl)Cl.C(=O)(C(F)(F)F)O>[CH3:1][CH2:2][CH2:3][CH2:4][CH2:5][CH2:6][CH2:7][CH2:8][CH2:9][CH2:10][CH2:11][CH2:12][CH2:13][CH2:14][CH2:15][CH2:16][CH2:17][C:18](=[O:19])Cl |f:1.5|'
        s4='O[C:18]([CH2:17][CH2:16][CH2:15][CH2:14][CH2:13][CH2:12][CH2:11][CH2:10][CH2:9][CH2:8][CH2:7][CH2:6][CH2:5][CH2:4][CH2:3][CH2:2][CH3:1])=[O:19]>CCCCCCCCCCCCCCCCCC(=O)O[C@H](C1=CC=CC=C1)C(=O)OC2=CC[C@]3([C@H]4CC5=CC=C(C6=C5[C@]3([C@H]2O6)CCN4C)O)O.CN(C)C=O.C1CCOC1.C(=O)(C(=O)Cl)Cl.C(=O)(C(F)(F)F)O>Cl[C:18]([CH2:17][CH2:16][CH2:15][CH2:14][CH2:13][CH2:12][CH2:11][CH2:10][CH2:9][CH2:8][CH2:7][CH2:6][CH2:5][CH2:4][CH2:3][CH2:2][CH3:1])=[O:19]'
        f2 = graph_util.rxn2features(s2)
        f3 = graph_util.rxn2features(s3)
        f4 = graph_util.rxn2features(s4)
        s = graph_util.feature_statistics([f2,f3])
        atom, bond1, bond2, conn1, conn2 = graph_util.encode_features_atommapped(f3, s, isrand=True)
        self.assertEqual(graph_util.build_mol_from_encoding(atom, bond1, conn1, s), 'CCCCCCCCCCCCCCCCCC(=O)O.Cl')
        self.assertEqual(graph_util.build_mol_from_encoding(atom, bond2, conn2, s), 'CCCCCCCCCCCCCCCCCC(=O)Cl.O')
        atom, bond1, bond2, conn1, conn2 = graph_util.encode_features_atommapped(f4, s, isrand=False)
        self.assertEqual(graph_util.build_mol_from_encoding(atom, bond1, conn1, s), 'CCCCCCCCCCCCCCCCCC(=O)O.Cl')
        self.assertEqual(graph_util.build_mol_from_encoding(atom, bond2, conn2, s), 'CCCCCCCCCCCCCCCCCC(=O)Cl.O')

    def test_encode_features_non_mapped(self):
        s2='[CH3:12][C:11]([CH3:13])([CH3:14])[O:10][C:8](=[O:9])[N:3]1[CH2:2][CH2:1][CH:6]([CH2:5][C:4]1=[O:7])OS(=O)(=O)C.[CH:20]1=[C:19]([N:18]=[CH:17][C:16]2=[C:15]1[CH:24]=[N:23][NH:22]2)[Cl:21]>>[CH3:12][C:11]([CH3:13])([CH3:14])[O:10][C:8](=[O:9])[N:3]1[CH2:2][CH2:1][CH:6]([CH2:5][C:4]1=[O:7])[N:22]2[C:16]3=[C:15]([CH:20]=[C:19]([N:18]=[CH:17]3)[Cl:21])[CH:24]=[N:23]2'
        s3='[CH3:1][CH2:2][CH2:3][CH2:4][CH2:5][CH2:6][CH2:7][CH2:8][CH2:9][CH2:10][CH2:11][CH2:12][CH2:13][CH2:14][CH2:15][CH2:16][CH2:17][C:18](=[O:19])O>CCCCCCCCCCCCCCCCCC(=O)O[C@H](C1=CC=CC=C1)C(=O)OC2=CC[C@]3([C@H]4CC5=CC=C(C6=C5[C@]3([C@H]2O6)CCN4C)O)O.CN(C)C=O.C1CCOC1.C(=O)(C(=O)Cl)Cl.C(=O)(C(F)(F)F)O>[CH3:1][CH2:2][CH2:3][CH2:4][CH2:5][CH2:6][CH2:7][CH2:8][CH2:9][CH2:10][CH2:11][CH2:12][CH2:13][CH2:14][CH2:15][CH2:16][CH2:17][C:18](=[O:19])Cl |f:1.5|'
        f2 = graph_util.rxn2features(s2)
        f3 = graph_util.rxn2features(s3)
        s = graph_util.feature_statistics([f2,f3])
        atom1, bond1, conn1, atom2, bond2, conn2 = graph_util.encode_features_non_mapped(f3, s, isrand=True)
        self.assertEqual(graph_util.build_mol_from_encoding(atom1, bond1, conn1, s), 'CCCCCCCCCCCCCCCCCC(=O)O')
        self.assertEqual(graph_util.build_mol_from_encoding(atom2, bond2, conn2, s), 'CCCCCCCCCCCCCCCCCC(=O)Cl')

if __name__ == '__main__':
    res = unittest.main(verbosity=3, exit=False)
