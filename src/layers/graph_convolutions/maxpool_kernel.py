from src.utils.inits import *
from src.layers.graph_convolutions.kernel import Kernels_new
# TODO Chebyshev style


class Kernel(Kernels_new):

    def __init__(self, **kwargs):
        super(Kernel, self).__init__(**kwargs)

        self.vars['weights_pool'] = glorot(shape=[self.input_dims['h'], self.input_dims['h']], name='weights_pool')
        self.vars['bias_pool'] = zeros([self.input_dims['h']], name='bias_pool')
        if self.bias:
            self.vars['bias'] = zeros([self.output_dim], name='bias')

    def _call(self, inputs):
        # DROPOUT
        data = {}
        data['x'] = inputs['activations'][1]
        data['h'] = inputs['activations'][-1]
        if self.add_labels:
            data['l'] = inputs['labels']

        self.node = self.compute_node_features(data, self.weights_node, self.bias_node, inputs['n_conn_nodes'])
        if self.shared_weights:
            self.neighbor = self.compute_neigh_features(data, self.weights_node, self.bias_node, inputs['adjmat'],
                                                        inputs['degrees'], inputs['n_conn_nodes'])
        else:
            self.neighbor = self.compute_neigh_features(data, self.weights_neigh, self.bias_neigh, inputs['adjmat'],
                                                        inputs['degrees'], inputs['n_conn_nodes'])

        self.get_gating_values(inputs['degrees'])
        h = self.combine()

        if self.skip_connetion:
            h = h + inputs['activations'][-1]
        if self.bias:
            h += self.vars['bias']

        h = self.act(tf.squeeze(h))
        # h = tf.nn.l2_normalize(h, dim=1)
        return h

    def compute_node_features(self, data, weights, bias, n_nodes):
        return self.compute_features(data, weights, bias, self.node_feautures, n_nodes)

    def compute_neigh_features(self, data, weights, bias, adjmat, degrees, n_nodes):
        data['h'] = self.combine_neighbor_info(adjmat, degrees, n_nodes, data['h'])
        neighbors = self.compute_features(data, weights, bias, self.neighbor_features, n_nodes)
        return neighbors

    def combine_neighbor_info(self, adjmat, degrees, n_nodes, h):
        n_nodes = tf.cast(n_nodes, tf.int32)
        adjmat = tf.sparse_tensor_to_dense(adjmat, validate_indices=False)
        h = tf.tile(tf.expand_dims(tf.squeeze(h), axis=0), multiples=(n_nodes, 1, 1))
        adjmat = tf.tile(tf.expand_dims(adjmat, axis=2), multiples=(1, 1, self.input_dims['h']))
        h = tf.multiply(adjmat, h)
        h = tf.reduce_max(h, axis=1)
        # h = tf.nn.relu(tf.matmul(h, self.vars['weights_pool']) + self.vars['bias_pool'])
        return h

    def get_gating_values(self, degrees):
        self.g0 = tf.ones_like(degrees) * 1
        self.g1 = tf.ones_like(degrees) * 1
