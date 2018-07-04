from src.utils.utils import *
from src.utils.inits import *
from src.layers.graph_convolutions.kernel import Kernels_new


class Kernel(Kernels_new):

    def __init__(self, **kwargs):
        super(Kernel, self).__init__(**kwargs)
        if self.bias:
            self.vars['bias'] = zeros([self.output_dim], name='bias')

    def _call(self, inputs):

        data = {}
        scaled_adjmat = inputs['adjmat']
        data['h_k-1'] = inputs['activations'][-1]

        if self.layer_id < 2:
            data['h'] = tf.sparse_tensor_dense_matmul(scaled_adjmat, data['h_k-1'])
        else:
            data['h_k-2'] = inputs['activations'][-2]
            data['h'] = 2*tf.sparse_tensor_dense_matmul(scaled_adjmat, data['h_k-1']) - data['h_k-2']

        h = self.compute_features(data, self.weights_node, self.bias_node, self.node_feautures, inputs['n_conn_nodes'])
        if self.bias:
            h += self.vars['bias']

        return h


