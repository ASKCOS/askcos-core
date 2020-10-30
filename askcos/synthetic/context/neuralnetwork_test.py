import os
import pickle
import unittest

import numpy as np

import askcos.global_config as gc
import askcos.synthetic.context.neuralnetwork as nn


class TestNeuralNetwork(unittest.TestCase):

    def test_01_get_n_conditions(self):
        """Test that NeuralNetContextRecommender.get_n_conditions gives the expected result."""
        cont = nn.NeuralNetContextRecommender()
        cont.load_nn_model(
            model_path=gc.NEURALNET_CONTEXT_REC['model_path'],
            info_path=gc.NEURALNET_CONTEXT_REC['info_path'],
            weights_path=gc.NEURALNET_CONTEXT_REC['weights_path']
        )
        result, scores = cont.get_n_conditions('CC1(C)OBOC1(C)C.Cc1ccc(Br)cc1>>Cc1cccc(B2OC(C)(C)C(C)(C)O2)c1', 10,
                                               with_smiles=False, return_scores=True)

        nan = float('nan')
        expected_result = [
            [102.30387878417969, 'C1COCCO1', 'CCN(CC)CC', "Reaxys Name (1,1'-bis(diphenylphosphino)ferrocene)palladium(II) dichloride", nan, nan, None, False],
            [104.92787170410156, 'C1COCCO1', 'CCN(CC)CC', 'Cl[Pd](Cl)([P](c1ccccc1)(c1ccccc1)c1ccccc1)[P](c1ccccc1)(c1ccccc1)c1ccccc1', nan, nan, None, False],
            [99.1409912109375, 'Cc1ccccc1', 'CCN(CC)CC', 'Cl[Pd](Cl)([P](c1ccccc1)(c1ccccc1)c1ccccc1)[P](c1ccccc1)(c1ccccc1)c1ccccc1', nan, nan, None, False],
            [76.38555908203125, 'C1CCOC1', 'CCN(CC)CC', 'Cl[Pd](Cl)([P](c1ccccc1)(c1ccccc1)c1ccccc1)[P](c1ccccc1)(c1ccccc1)c1ccccc1', nan, nan, None, False],
            [95.92562103271484, 'Cc1ccccc1', 'CCN(CC)CC', "Reaxys Name (1,1'-bis(diphenylphosphino)ferrocene)palladium(II) dichloride", nan, nan, None, False],
            [75.68882751464844, 'C1CCOC1', 'CCN(CC)CC', "Reaxys Name (1,1'-bis(diphenylphosphino)ferrocene)palladium(II) dichloride", nan, nan, None, False],
            [93.39191436767578, 'C1COCCO1', '', "Reaxys Name (1,1'-bis(diphenylphosphino)ferrocene)palladium(II) dichloride", nan, nan, None, False],
            [97.8741226196289, 'C1COCCO1', 'CC(=O)[O-].[K+]', "Reaxys Name (1,1'-bis(diphenylphosphino)ferrocene)palladium(II) dichloride", nan, nan, None, False],
            [95.84452819824219, 'C1COCCO1', '[MgH2]', 'Cl[Pd](Cl)([P](c1ccccc1)(c1ccccc1)c1ccccc1)[P](c1ccccc1)(c1ccccc1)c1ccccc1', nan, nan, None, False],
            [67.86063385009766, 'C1CCOC1', '[MgH2]', 'Cl[Pd](Cl)([P](c1ccccc1)(c1ccccc1)c1ccccc1)[P](c1ccccc1)(c1ccccc1)c1ccccc1', nan, nan, None, False],
        ]
        expected_scores = [0.19758703, 0.09385002, 0.0320574, 0.026747962, 0.024693565, 0.010140889, 0.0048135854, 0.004163743, 0.002136398, 0.0018915414]

        for e, r in zip(expected_result, result):
            self.assertEqual(len(e), len(r))
            self.assertAlmostEqual(e[0], r[0], places=4)
            self.assertEqual(e[1:4], r[1:4])
            self.assertEqual(e[7:], r[7:])

        self.assertTrue(np.allclose(expected_scores, scores))


if __name__ == '__main__':
    res = unittest.main(verbosity=3, exit=False)
