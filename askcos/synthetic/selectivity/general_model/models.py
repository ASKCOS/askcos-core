import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K
from askcos.synthetic.selectivity.general_model.layers import WLN_Layer, Global_Attention


class WLNReactionClassifier(tf.keras.Model):
    '''

    '''
    def __init__(self, hidden_size, depth=4, max_nb=10):
        super(WLNReactionClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.reactants_WLN = WLN_Layer(hidden_size, depth, max_nb)
        self.products_WLN = WLN_Layer(hidden_size, depth, max_nb)
        self.attention = Global_Attention(hidden_size)

        self.reactant_feature = layers.Dense(hidden_size, activation=K.relu, kernel_initializer=tf.random_normal_initializer(stddev=0.1), use_bias=False)
        self.product_feature = layers.Dense(hidden_size, activation=K.relu, kernel_initializer=tf.random_normal_initializer(stddev=0.1), use_bias=False)

        self.reaction_score = layers.Dense(1, kernel_initializer=tf.random_normal_initializer(stddev=0.1))

        self.node_reshape = layers.Reshape((-1, 1))
        self.core_reshape = layers.Reshape((-1, 1))

    def call(self, inputs):
        res_inputs = inputs[:8]
        prods_inputs = inputs[8:]

        res_atom_mask = res_inputs[-3]
        prod_atom_mask = prods_inputs[-2]

        res_core_mask = res_inputs[-2]
        prod_core_mask = prods_inputs[-1]

        res_bin_features = res_inputs[-1]

        res_atom_hidden = self.reactants_WLN(res_inputs[:-1])
        res_att_context, _ = self.attention(res_atom_hidden, res_bin_features)
        res_atom_hidden = res_atom_hidden + res_att_context
        res_atom_hidden = K.relu(res_atom_hidden)
        res_atom_mask = self.node_reshape(res_atom_mask)
        res_core_mask = self.core_reshape(res_core_mask)
        res_mol_hidden = K.sum(res_atom_hidden*res_atom_mask*res_core_mask, axis=-2)
        res_mol_hidden = self.reactant_feature(res_mol_hidden)

        prod_atom_hidden = self.products_WLN(prods_inputs)
        prod_atom_hidden = K.relu(prod_atom_hidden)
        prod_atom_mask = self.node_reshape(prod_atom_mask)
        prod_core_mask = self.core_reshape(prod_core_mask)
        prod_mol_hidden = K.sum(prod_atom_hidden*prod_atom_mask*prod_core_mask, axis=-2)
        prod_mol_hidden = self.product_feature(prod_mol_hidden)

        reaction_hidden = tf.concat([prod_mol_hidden, res_mol_hidden], axis=-1)
        reaction_score = self.reaction_score(reaction_hidden)

        return reaction_score
