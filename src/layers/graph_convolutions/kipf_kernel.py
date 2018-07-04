from src.utils.utils import *
from src.utils.inits import *
from src.layers.graph_convolutions.kernel import Kernels_new


class Kernel(Kernels_new):

    def __init__(self, **kwargs):
        super(Kernel, self).__init__(**kwargs)
        if self.bias:
            self.vars['bias'] = zeros([self.output_dim], name='bias')

    def _call(self, inputs):
        # DROPOUT
        data = {}
        F_Adjmat = inputs['adjmat']
        data['h'] = inputs['activations'][-1]
        if self.layer_id > 0:
            data['x'] = inputs['activations'][1]
        else:
            data['x'] = data['h']

        if self.add_labels:
            data['l'] = inputs['labels']

        self.node = self.compute_node_features(data, self.weights_node, self.bias_node, inputs['n_conn_nodes'])
        if self.shared_weights:
            self.neighbor = self.compute_neigh_features(data, self.weights_node, self.bias_node, F_Adjmat,
                                                        inputs['n_conn_nodes'])
        else:
            self.neighbor = self.compute_neigh_features(data, self.weights_neigh, self.bias_neigh, F_Adjmat,
                                                        inputs['n_conn_nodes'])

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

    def compute_neigh_features(self, data, weights, bias, F_adjmat, n_nodes):
        if set(self.node_feautures) == set(self.neighbor_features):
            h = self.node
        else:
            h = self.compute_features(data, weights, bias, self.neighbor_features, n_nodes)
        neighbors = tf.sparse_tensor_dense_matmul(F_adjmat, h)
        return neighbors

    def get_gating_values(self, degrees):
        degrees = degrees + 1
        self.g0 = 1/degrees
        self.g1 = tf.ones_like(degrees)
