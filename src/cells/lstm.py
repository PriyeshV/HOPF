from src.utils.inits import *
from tensorflow.contrib.rnn import RNNCell


class MyLSTMCell(RNNCell):
    '''Vanilla LSTM implemented with same initializations as BN-LSTM'''

    def __init__(self, num_units, x_size, layer_id):
        self.num_units = num_units
        self.W_xh = tf.get_variable('W_xh'+layer_id, [x_size, 4 * self.num_units], initializer=orthogonal_initializer())
        self.W_hh = tf.get_variable('W_hh'+layer_id, [self.num_units, 4 * self.num_units],
                                    initializer=bn_lstm_identity_initializer(0.95))
        self.bias = tf.get_variable('bias'+layer_id, [4 * self.num_units])#, initializer=zeros)  # intializer ???

    def candidate_weights_bias(self):
        weights = tf.slice(self.W_xh, [0, 0], [-1, self.num_units])
        bias = self.bias[:][:self.num_units]
        return weights, bias

    @property
    def state_size(self):
        return tuple([self.num_units, self.num_units])

    @property
    def output_size(self):
        return self.num_units

    def __call__(self, x, c, h, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            # c, h = state

            # improve speed by concat.
            data = tf.concat([x, h], 1)  # shape = b*(d+h)
            W_both = tf.concat([self.W_xh, self.W_hh], 0)  # shape = (d+h)*4h
            hidden = tf.matmul(data, W_both) + self.bias  # shape = b*4h

            j, i, f, o = tf.split(hidden, axis=1, num_or_size_splits=4) #j and i changed

            new_c = c * tf.sigmoid(f) + tf.sigmoid(i) * tf.tanh(j)
            new_h = tf.tanh(new_c) * tf.sigmoid(o)

            return new_c, new_h