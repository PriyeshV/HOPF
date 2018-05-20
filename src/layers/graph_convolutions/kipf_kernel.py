from src.utils.utils import *
from src.utils.inits import *
from src.layers.graph_convolutions.kernel import Kernels_new
# TODO Chebyshev style


class Kernel(Kernels_new):

    def __init__(self, **kwargs):
        super(Kernel, self).__init__(**kwargs)
        if self.bias:
            self.vars['bias'] = zeros([self.output_dim], name='bias')

    def _call(self, inputs):
        # DROPOUT
        data = {}
        data['h'] = inputs['activations'][-1]
        if self.layer_id > 0:
            data['x'] = inputs['activations'][1]
        else:
            data['x'] = data['h']

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
        if self.bias:
            h += self.vars['bias']

        if self.layer_id == 0:
            return h, self.node
        else:
            return h

    def compute_node_features(self, data, weights, bias, n_nodes):
        return self.compute_features(data, weights, bias, self.node_feautures, n_nodes)

    def compute_neigh_features(self, data, weights, bias, adjmat, degrees, n_nodes):
        if set(self.node_feautures) == set(self.neighbor_features):
            h = self.node
        else:
            h = self.compute_features(data, weights, bias, self.neighbor_features, n_nodes)
        weights = self.combine_neighbor_info(adjmat, degrees, n_nodes)
        neighbors = tf.sparse_tensor_dense_matmul(weights, h)
        return neighbors

    def combine_neighbor_info(self, adjmat, degrees, n_nodes):
        if self.node_dims != 0:
            degrees = degrees + 1
        weights = self.get_laplacian(adjmat, degrees)
        # weights = self.get_laplacian2(adjmat, degrees)
        return weights

    def get_gating_values(self, degrees):
        degrees = degrees + 1
        self.g0 = 1/degrees
        self.g1 = tf.ones_like(degrees)

    def get_laplacian(self, adjmat, degrees):
        degrees = tf.expand_dims(1/tf.sqrt(degrees), axis=1)
        laplacian = adjmat.__mul__(degrees)
        degrees = tf.transpose(degrees, [1, 0])
        laplacian = laplacian.__mul__(degrees)
        return laplacian

    def get_laplacian2(self, adjmat, degrees):
        degrees = tf.expand_dims(1/degrees, axis=1)
        laplacian = adjmat.__mul__(degrees)
        return laplacian