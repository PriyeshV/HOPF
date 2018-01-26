from src.utils.utils import *
from src.utils.inits import *
from src.layers.graph_convolutions.kernel import Kernels_new
# TODO Chebyshev style


class Kernel(Kernels_new):

    def __init__(self, **kwargs):
        super(Kernel, self).__init__(**kwargs)

        # Define attention weights
        self.att_size = self.output_dim
        self.att_context_dim = self.output_dim
        self.att_input_dim = self.output_dim

        # self.vars['weights_att_X'] = sigmoid_init(shape=[self.att_input_dim, self.att_size], name='weights_att_x')
        # self.vars['weights_att_C'] = sigmoid_init(shape=[self.att_context_dim, self.att_size], name='weights_att_C')
        # self.vars['weights_att_V'] = sigmoid_init(shape=[self.att_size, 1], name='weights_att_V')
        self.vars['weights_att_X'] = glorot(shape=[self.att_input_dim, self.att_size], name='weights_att_x')
        self.vars['weights_att_C'] = glorot(shape=[self.att_context_dim, self.att_size], name='weights_att_C')
        self.vars['weights_att_V'] = glorot([self.att_size, 1], name='weights_att_V')

        self.vars['bias_att'] = zeros([1, 1, self.output_dim], name='bias_att')
        self.vars['bias_att1'] = zeros([self.output_dim], name='bias_att1')
        self.vars['bias_att2'] = zeros([self.output_dim], name='bias_att2')

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

        # h = self.act(tf.squeeze(h))
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

    def get_laplacian(self, adjmat, degrees):
        degrees = tf.expand_dims(1/degrees, axis=1)
        laplacian = adjmat.__mul__(degrees)
        return laplacian

    def get_attention_scores(self, adjmat, context, inputs, degrees, n_nodes):
        n_nodes = tf.cast(n_nodes, tf.int32)
        # context = tf.nn.sigmoid(context + self.vars['bias_att1'])
        # inputs = tf.nn.sigmoid(inputs + self.vars['bias_att2'])

        # context = tf.matmul(context,  self.vars['weights_att_C']) # + self.vars['bias_att1']
        # inputs = tf.matmul(inputs, self.vars['weights_att_X'])  # + self.vars['bias_att2']
        #
        # inputs = tf.reshape(inputs, [n_nodes, 1, self.att_size])
        # context = tf.reshape(context, [n_nodes, 1, self.att_size])
        # attn_matrix = tf.nn.tanh(inputs + tf.transpose(context, [1, 0, 2]))  # [path,1,A]+[1,path,A] -> [path, path, A]
        # if self.bias:
        #     attn_matrix += self.vars['bias_att']
        #
        # attn_matrix = tf.reshape(attn_matrix, [n_nodes * n_nodes, self.att_size])
        # attn_matrix = tf.matmul(attn_matrix, self.vars['weights_att_V'])
        # attn_matrix = tf.reshape(attn_matrix, [n_nodes, n_nodes])
        # scores = adjmat.__mul__(attn_matrix)
        # scores = tf.sparse_softmax(scores)
        scores = adjmat
        return scores
