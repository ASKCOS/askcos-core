import os
import pickle
import unittest

import askcos.synthetic.evaluation.rexgen_release.predict as p


@unittest.skip('Models have been removed.')
class TestPredict(unittest.TestCase):

    def test_01_predict(self):
        """Test template free forward predictor."""
        tffp = p.TFFP()
        result = tffp.predict('CCCO.CCCBr')

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
