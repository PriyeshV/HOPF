from src.layers.layer import Layer
from src.utils.utils import *
from src.utils.inits import *
import tensorflow as tf

class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, nnz_features, dropout, sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        self.dropout = dropout

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.nnz_features = nnz_features

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim], name='weights')
            self.var_reassign = tf.assign(self.vars['weights'], glorot([input_dim, output_dim], name='weights'))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs['activations'][-1]

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.nnz_features)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # transform
        h = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            h += self.vars['bias']

        # h = self.act(h)
        # h = tf.nn.l2_normalize(h, dim=1)
        return h
