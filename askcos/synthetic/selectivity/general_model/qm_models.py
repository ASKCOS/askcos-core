import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K

from .qm_layers import WLN_Layer, Global_Attention

class QMWLNPairwiseAtomClassifier(tf.keras.Model):
    '''

    '''

    def __init__(self, hidden_size, qm_size=160, depth=4, max_nb=10):
        super(QMWLNPairwiseAtomClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.qm_size = qm_size
        self.reactants_WLN = WLN_Layer(hidden_size, depth, max_nb)
        self.reagents_WLN = WLN_Layer(hidden_size, depth, max_nb)
        self.attention = Global_Attention(hidden_size)

        self.reaction_score0 = layers.Dense(2*hidden_size+qm_size+10, activation=K.relu, kernel_initializer=tf.random_normal_initializer(stddev=0.01), use_bias=False)

        self.reaction_score = layers.Dense(1, kernel_initializer=tf.random_normal_initializer(stddev=0.01), use_bias=False)

        self.reshape1 = layers.Reshape((1, -1, 2*hidden_size+qm_size))
        self.reshape2 = layers.Reshape((-1, 1, 2*hidden_size+qm_size))

    def call(self, inputs):
        res_inputs = inputs[:6]
        res_core_mask = inputs[6][0]
        fatom_qm = inputs[7]
        connect = inputs[8][0]

        rg_inputs = inputs[9:]

        #machine learned representation
        res_atom_hidden = self.reactants_WLN(res_inputs)
        rg_atom_hidden = self.reagents_WLN(rg_inputs)

        res_att_context = self.attention(res_atom_hidden, rg_atom_hidden)

        res_atom_hidden = K.concatenate([res_atom_hidden, res_att_context, fatom_qm], axis=-1)

        #select out reacting cores
        atom_hiddens1 = self.reshape1(res_atom_hidden)
        atom_hiddens2 = self.reshape2(res_atom_hidden)
        atom_pair = atom_hiddens1 + atom_hiddens2

        atom_pair = tf.gather_nd(atom_pair, res_core_mask)
        reaction_hidden = K.concatenate([atom_pair, connect], axis=-1)
        reaction_hidden = self.reaction_score0(reaction_hidden)

        reaction_seg = res_core_mask[:, 0]
        reaction_hidden = tf.math.segment_mean(reaction_hidden, reaction_seg)
        reaction_score = self.reaction_score(reaction_hidden)
        return reaction_score


class WLNPairwiseAtomClassifierNoReagent(tf.keras.Model):
    '''

    '''

    def __init__(self, hidden_size, qm_size=160, depth=4, max_nb=10):
        super(WLNPairwiseAtomClassifierNoReagent, self).__init__()
        self.hidden_size = hidden_size
        self.qm_size = qm_size
        self.reactants_WLN = WLN_Layer(hidden_size, depth, max_nb)
        self.reaction_score0 = layers.Dense(hidden_size+qm_size+10, activation=K.relu, kernel_initializer=tf.random_normal_initializer(stddev=0.01), use_bias=False)

        self.reaction_score = layers.Dense(1, kernel_initializer=tf.random_normal_initializer(stddev=0.01), use_bias=False)

        self.reshape1 = layers.Reshape((1, -1, hidden_size+qm_size))
        self.reshape2 = layers.Reshape((-1, 1, hidden_size+qm_size))

    def call(self, inputs):
        res_inputs = inputs[:6]
        res_core_mask = inputs[6][0]
        fatom_qm = inputs[7]
        connect = inputs[8][0]

        rg_inputs = inputs[9:]

        #machine learned representation
        res_atom_hidden = self.reactants_WLN(res_inputs)

        res_atom_hidden = K.concatenate([res_atom_hidden, fatom_qm], axis=-1)
        #select out reacting cores
        atom_hiddens1 = self.reshape1(res_atom_hidden)
        atom_hiddens2 = self.reshape2(res_atom_hidden)
        atom_pair = atom_hiddens1 + atom_hiddens2

        atom_pair = tf.gather_nd(atom_pair, res_core_mask)
        reaction_hidden = K.concatenate([atom_pair, connect], axis=-1)
        reaction_hidden = self.reaction_score0(reaction_hidden)

        reaction_seg = res_core_mask[:, 0]
        reaction_hidden = tf.math.segment_mean(reaction_hidden, reaction_seg)
        reaction_score = self.reaction_score(reaction_hidden)
        return reaction_score

