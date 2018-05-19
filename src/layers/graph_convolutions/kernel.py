import tensorflow as tf
from src.layers.layer import Layer
from src.utils.inits import glorot, const
from src.utils.utils import sparse_dropout, dot

class Kernels_new(Layer):

    def __init__(self, layer_id, x_names, dims, dropout, act=tf.nn.relu, nnz_features=None, sparse_inputs=False,
                 bias=False, shared_weights=True, skip_connection=False, add_labels=False, **kwargs):
        super(Kernels_new, self).__init__(**kwargs)

        self.layer_id = layer_id
        self.act = act
        self.dropout = dropout
        self.skip_connetion = skip_connection
        self.add_labels = add_labels
        self.nnz_features = nnz_features
        self.sparse_inputs = sparse_inputs

        self.bias = bias
        self.shared_weights = shared_weights
        self.node = None
        self.neighbor = None
        self.g0 = None
        self.g1 = None

        self.node_feautures = x_names[0]
        self.neighbor_features = x_names[1]
        if self.add_labels:
            if len(self.neighbor_features) == 0:
                self.neighbor_features = ['l']
            else:
                self.neighbor_features.append('l')

        # Weights initialization
        self.input_dims = {}
        # Initially 'x' and 'h' save same input
        if self.layer_id == 0 and self.m_name != '':
            self.input_dims['x'] = dims[0]
        else:
            self.input_dims['x'] = dims[1]

        self.input_dims['h'] = dims[layer_id]
        self.input_dims['l'] = dims[-1]
        self.output_dim = dims[layer_id+1]

        self.node_dims = 0
        self.neigh_dims = 0

        if self.bias:
            self.vars['bias'] = const([self.output_dim])

        # Compute total dimensions
        for key in self.node_feautures:
            self.node_dims += self.input_dims[key]
        for key in self.neighbor_features:
            self.neigh_dims += self.input_dims[key]

        if not shared_weights:
            # Neigh weights
            self.weights_neigh = {}
            self.bias_neigh = None
            with tf.variable_scope(self.name + "_neighbor_vars"):
                keys = self.neighbor_features
                for key in keys:
                    self.weights_neigh[key] = glorot((self.input_dims[key], self.output_dim), name=key + 'weights')

        # Node weights
        with tf.variable_scope(self.name + "_node_vars"):
            self.weights_node = {}
            self.bias_node = None
            if shared_weights:
                keys = self.node_feautures + list(set(self.neighbor_features) - set(self.node_feautures))
            else:
                keys = self.node_feautures
            for key in keys:
                self.weights_node[key] = glorot((self.input_dims[key], self.output_dim), name=key+'weights')

    def compute_features(self, inputs, weights, bias, keys, n_nodes):
            if len(keys) == 0:
                return tf.zeros(shape=(tf.cast(n_nodes, dtype=tf.int32), self.output_dim))
            output = tf.zeros(shape=(self.output_dim), dtype=tf.float32)
            for key in keys:
                data = inputs[key]
                dropout = self.dropout
                if key == 'l':  # and self.layer_id == 0:
                    dropout = tf.minimum(self.dropout, 0.)
                    sparse_inputs = False
                else:
                    sparse_inputs = self.sparse_inputs
                if not self.sparse_inputs or key == 'l':
                    data = tf.nn.dropout(data, 1 - dropout)
                else:
                    data = sparse_dropout(data, 1 - dropout, self.nnz_features)
                output += dot(data, weights[key], sparse=sparse_inputs)
            return output

    def combine(self):
        node = tf.multiply(tf.expand_dims(self.g0, axis=1), self.node)
        neighbor = tf.multiply(tf.expand_dims(self.g1, axis=1), self.neighbor)
        return node + neighbor


class Kernels(Layer):

    def __init__(self, **kwargs):
        super(Kernels, self).__init__(**kwargs)

    def combine(self, g0, g1, node, neighbors, node_W, neigh_W):
        node = tf.matmul(tf.multiply(tf.expand_dims(g0, axis=0), node), node_W)
        neighbors = tf.matmul(tf.multiply(tf.expand_dims(g1, axis=0), neighbors), neigh_W)
        return node + neighbors