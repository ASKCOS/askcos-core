import unittest

import tensorflow.compat.v1 as tf

from askcos.synthetic.selectivity.general_selectivity import GeneralSelectivityPredictor

class GeneralSelectivity(unittest.TestCase):

    def setUp(self) -> None:
        """This method is run once before each test in this class"""
        tf.keras.backend.clear_session()

    def test_qm_gnn_predictor(self):
        "Test qm_gnn predictor"
        pass