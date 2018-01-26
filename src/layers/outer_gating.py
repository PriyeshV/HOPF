import tensorflow as tf
from src.layers.layer import Layer
from src.utils.inits import glorot


class gated_prediction(Layer):

    def __init__(self, n_layers, x_names, dims, dropout, values, bias, **kwargs):
        super(gated_prediction, self).__init__(**kwargs)

        self.n_layers = n_layers + 1
        self.output_dim = dims[-1]
        self.dropout = dropout
        self.node_features = x_names[0]
        self.neighbor_features = x_names[1]

        self.values = values
        self.bias = bias

        self.start_h = 0
        if len(self.node_features) == 0:
            self.start_h = 1

        dims[0] = dims[1]
        for i in range(self.start_h, self.n_layers):
            self.vars['weights_'+str(i)] = glorot((dims[i], self.output_dim), name='weights_'+str(i))

    def _call(self, inputs):
        outputs = []
        for i in range(self.start_h, self.n_layers):
            data = tf.nn.dropout(inputs['activations'][i+1], 1 - self.dropout)
            outputs.append(tf.matmul(data, self.vars['weights_'+str(i)]))
        gate_input = tf.concat(outputs, axis=1)
        outputs = tf.squeeze(tf.reduce_sum(outputs, axis=0))
        return outputs, gate_input

