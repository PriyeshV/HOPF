from src.layers.layer import Layer
from src.utils.inits import *
from src.utils.utils import *


class RegularizerPos(Layer):

    def __init__(self, config, M, **kwargs):
        super(RegularizerPos, self).__init__(**kwargs)

        self.alpha = config.regKernel['var']
        self.K = config.regKernel['order']
        # self.vars['M'] = M

    def _call(self, inputs):
        embeddings = inputs['embeddings']
        adjmat = tf.sparse_to_dense(inputs['adjmat'].indices, inputs['adjmat'].dense_shape, inputs['adjmat'].values, validate_indices=False)
        degrees = tf.diag(tf.reduce_sum(adjmat, axis=1) + tf.constant(1e-15))
        L = degrees - adjmat
        L_k = L
        # L_k = tf.matmul(tf.pow(degrees, -0.5), tf.transpose(tf.matmul(tf.pow(degrees, -0.5), L), [1, 0]))

        for k in range(self.K-1):
            L_k = tf.matmul(L_k, L)

        # TODO Residual R = L + Attention matrix
        R = L_k

        loss = 1 - tf.exp(-1 * tf.trace(tf.matmul(tf.matmul(tf.transpose(embeddings, [1, 0]), R), embeddings))/tf.trace(R))
        return loss, L_k


class RegularizerNeg(Layer):

    def __init__(self, config, M, **kwargs):
        super(RegularizerNeg, self).__init__(**kwargs)

        self.alpha = config.regKernel['var']
        self.K = config.regKernel['order']
        # self.vars['M'] = M

    def _call(self, inputs):
        # D-(I + D+ + D+D+ + ...) - A-(I + A+ + A+A+ )
        L_k = inputs['L_k']
        zero = tf.constant(0, dtype=tf.float32)
        where_pos = tf.equal(L_k, zero)
        A_ = tf.cast(where_pos, dtype=tf.float32)
        D_ = tf.diag(tf.reduce_sum(A_, axis=1) + tf.constant(1e-15))
        L_ = D_ - A_

        embeddings = inputs['embeddings']
        L_k = tf.matmul(L_, L_k)

        # TODO Residual
        R = L_k

        loss = tf.exp(-1 * tf.trace(tf.matmul(tf.matmul(tf.transpose(embeddings, [1, 0]), R), embeddings))/tf.trace(R))
        return loss
