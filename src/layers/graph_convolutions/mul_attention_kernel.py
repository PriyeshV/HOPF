from src.utils.utils import *
from src.utils.inits import *
from src.layers.graph_convolutions.kernel import Kernels_new
# TODO Chebyshev style


class Kernel(Kernels_new):

    def __init__(self, **kwargs):
        super(Kernel, self).__init__(**kwargs)

        self.vars['weights_att'] = glorot([self.output_dim, self.output_dim], name='weights_att')
        self.vars['bias_att1'] = zeros(self.output_dim, name='bias_att1')
        self.vars['bias_att2'] = zeros(self.output_dim, name='bias_att2')
        self.vars['bias_att'] = const([1], name='bias_att')
        self.vars['deg_bias'] = ones(1, name='deg_bias')
        if self.bias:
            self.vars['bias'] = zeros([self.output_dim], name='bias')

    def _call(self, inputs):
        # DROPOUT
        data = {}
        x = inputs['activations'][1]
        h = inputs['activations'][-1]
        data['x'] = x
        data['h'] = h
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
        h = self.compute_features(data, weights, bias, self.neighbor_features, n_nodes)
        neighbors = self.combine_neighbor_info(adjmat, degrees, n_nodes, h)
        return neighbors

    def combine_neighbor_info(self, adjmat, degrees, n_nodes, h):
        weights = self.get_attention_scores(adjmat, self.node, h, degrees, n_nodes)
        neighbors = tf.sparse_tensor_dense_matmul(weights, h)
        return neighbors

    def get_gating_values(self, degrees):
        # self.g0 = 1/degrees
        self.g0 = tf.ones_like(degrees) * 1
        self.g1 = tf.ones_like(degrees) * 1

    def get_attention_scores(self, adjmat, context, inputs, degrees, n_nodes):
        context = tf.nn.sigmoid(context + self.vars['bias_att1'])
        inputs = tf.nn.sigmoid(inputs + self.vars['bias_att2'])

        scores = tf.nn.tanh(tf.matmul(tf.matmul(context, self.vars['weights_att']), tf.transpose(inputs, [1, 0])))
        scores = adjmat.__mul__(scores)
        attention_scores = tf.sparse_softmax(scores)
        return attention_scores

