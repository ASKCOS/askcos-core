import tensorflow as tf
from tensorflow.keras import backend as K


def wln_loss(y_true, y_pred):

    flat_label = K.cast(K.reshape(y_true, [-1]), 'float32')
    flat_score = K.reshape(y_pred, [-1])

    reaction_seg = K.cast(tf.math.cumsum(flat_label), 'int32') - tf.constant([1], dtype='int32')

    max_seg = tf.gather(tf.math.segment_max(flat_score, reaction_seg), reaction_seg)
    exp_score = tf.exp(flat_score-max_seg)

    softmax_denominator = tf.gather(tf.math.segment_sum(exp_score, reaction_seg), reaction_seg)
    softmax_score = exp_score/softmax_denominator

    softmax_score = tf.clip_by_value(softmax_score, K.epsilon(), 1-K.epsilon())

    try:
        loss =  -tf.reduce_sum(flat_label * tf.math.log(softmax_score))/flat_score.shape[0]
        return loss
    except:
        return -tf.reduce_sum(flat_label * tf.math.log(softmax_score))