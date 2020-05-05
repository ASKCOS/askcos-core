from functools import partial

import tensorflow.compat.v1 as tf
from scipy.special import softmax

from makeit import global_config as gc
from makeit.synthetic.selectivity.general_model.data_loading import GraphDataLoader
from makeit.synthetic.selectivity.general_model.loss import wln_loss
from makeit.synthetic.selectivity.general_model.models import WLNReactionClassifier
from makeit.synthetic.selectivity.mol_graph import atom_fdim as adim, bond_fdim as bdim, max_nb, \
    smiles2graph_pr as _s2g

model_path = gc.GEN_SELECTIVITY['model_path']
initializer = gc.GEN_SELECTIVITY['initializer']


class GeneralSelectivityPredictor:

    def __init__(self, depth=4, hidden_size=200, batch_size=1, core_buffer=2):
        
        self.depth = depth
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.save_path = model_path
        self.smiles2graph_batch = partial(_s2g, idxfunc=lambda x: x.GetIdx(), core_buffer=core_buffer)
        self.adim = adim
        self.bdim = bdim
        self.max_nb = max_nb
        self.DataLoader = partial(GraphDataLoader, batch_size=self.batch_size, shuffle=False)
        self.graph = tf.get_default_graph()

        self.model = self.build()
        
    def build(self):

        initializer_x, _ = self.DataLoader([initializer.split('>')[0]], [initializer.split('>')[2]])[0]

        model = WLNReactionClassifier(self.hidden_size, self.depth, self.max_nb)
        opt = tf.keras.optimizers.Adam(lr=0.001, clipnorm=5.0)
        model.compile(
            optimizer=opt,
            loss=wln_loss,
        )
        model.predict_on_batch(initializer_x)
        model.load_weights(model_path)

        return model

    def predict(self, smiles):
        '''
        predict the selectivity from atommapped reactions as reactants>>product1.product2.product3....
        :param smiles:
        :param template:
        :return:
        '''

        #FIXME the dynamic model seems like to be incompatible with the celery, that the model created in build
        # actaully cannot be reused in the predict, currently an akward method is used to create and load model each
        # time the predict function is called. The grapn network need to be modified in to static graph later.

        reactants, _, products = smiles.split('>')

        initializer_x, _ = self.DataLoader([initializer.split('>')[0]], [initializer.split('>')[2]])[0]

        model = WLNReactionClassifier(self.hidden_size, self.depth, self.max_nb)
        opt = tf.keras.optimizers.Adam(lr=0.001, clipnorm=5.0)
        model.compile(
            optimizer=opt,
            loss=wln_loss,
        )
        model.predict_on_batch(initializer_x)
        model.load_weights(model_path)

        self.model = model

        test_gen = self.DataLoader([reactants], [products])

        batch_input, _ = test_gen[0]
        out = self.model.predict_on_batch(batch_input).reshape([-1])
        out = tuple([float(x) for x in softmax(out)])

        return out


# for testing purposes
if __name__ == "__main__":
    predictor = GeneralSelectivityPredictor()
    react = '[Br:1][Br:2].[NH2:3][c:4]1[n:5][cH:6][n:7][c:8]2[nH:9][cH:10][n:11][c:12]12>O>' \
            '[Br:2][c:10]1[nH:9][c:8]2[n:7][cH:6][n:5][c:4]([NH2:3])[c:12]2[n:11]1.' \
            '[Br:2][c:6]1[n:5][c:4]([NH2:3])[c:12]2[c:8]([n:7]1)[nH:9][cH:10][n:11]2'
    print(react)
    res = predictor.predict(react)
    print(res)
