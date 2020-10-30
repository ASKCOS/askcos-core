import unittest

from askcos.synthetic.selectivity.general_selectivity import QmGnnGeneralSelectivityPredictor, \
    QmGnnGeneralSelectivityPredictorNoReagent, GnnGeneralSelectivityPredictor

class GeneralSelectivity(unittest.TestCase):

    def test_qm_gnn_predictor_reagents(self):
        """Test qm_gnn predictor"""
        rxn = 'CC(COc1n[nH]cc1)C.CC(C)(OC(c1c(Cl)nc(Cl)cc1)=O)C>CN(C=O)C.O>CC(OC(c1ccc(n2ccc(OCC(C)C)n2)nc1Cl)=O)(C)C'
        predictor = QmGnnGeneralSelectivityPredictor()

        res = predictor.predict(rxn)
        self.assertEqual(len(res), 2)
        self.assertEqual(type(res[0]), dict)
        self.assertAlmostEqual(res[0]['prob'], 1, 2)

    def test_qm_gnn_predictor_no_reagents(self):
        """Test qm_gnn predictor"""
        rxn = 'CC(COc1n[nH]cc1)C.CC(C)(OC(c1c(Cl)nc(Cl)cc1)=O)C>>CC(OC(c1ccc(n2ccc(OCC(C)C)n2)nc1Cl)=O)(C)C'
        predictor = QmGnnGeneralSelectivityPredictorNoReagent()

        res = predictor.predict(rxn)
        self.assertEqual(len(res), 2)
        self.assertEqual(type(res[0]), dict)
        self.assertAlmostEqual(res[0]['prob'], 1, 2)

    def test_gnn_predictor(self):
        """Test qm_gnn predictor"""
        rxn = 'CC(COc1n[nH]cc1)C.CC(C)(OC(c1c(Cl)nc(Cl)cc1)=O)C>>CC(OC(c1ccc(n2ccc(OCC(C)C)n2)nc1Cl)=O)(C)C'
        predictor = GnnGeneralSelectivityPredictor()

        res = predictor.predict(rxn)
        self.assertEqual(len(res), 2)
        self.assertEqual(type(res[0]), dict)
        self.assertAlmostEqual(res[0]['prob'], 1, 2)


if __name__ == '__main__':
    res = unittest.main(verbosity=3, exit=False)
