import os
import pickle
import unittest

import askcos.synthetic.evaluation.rexgen_direct.predict as p


class TestPredict(unittest.TestCase):

    def test_01_predict(self):
        """Test template free forward predictor."""
        tffp = p.TFFP()
        rsmi_am, result = tffp.predict('CCCO.CCCBr')

        self.assertEqual('[CH3:1][CH2:2][CH2:3][OH:4].[CH3:5][CH2:6][CH2:7][Br:8]', rsmi_am)

        with open(os.path.join(os.path.dirname(__file__), 'test_data/predict.pkl'), 'rb') as t:
            expected = pickle.load(t, encoding='iso-8859-1')

        self.assertEqual(len(expected), len(result))

        for e, r in zip(expected, result):
            self.assertEqual(e['smiles'], r['smiles'])
            self.assertEqual(e['rank'], r['rank'])
            self.assertAlmostEqual(e['prob'], r['prob'], places=4)
            self.assertAlmostEqual(e['score'], r['score'], places=4)


if __name__ == '__main__':
    res = unittest.main(verbosity=3, exit=False)
