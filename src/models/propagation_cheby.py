from src.models.model import Model
from src.utils.metrics import *
from src.layers.dense import Dense
from src.layers.fusion_weighted_sum import Fusion

class Propagation(Model):

    def __init__(self, config, data,  **kwargs):
        kwargs['name'] = 'cheby'
        super(Propagation, self).__init__(**kwargs)

        self.data.update(data)
        self.inputs = data['features']
        self.l2 = config.l2
        self.add_labels = config.add_labels
        self.bias = config.bias
        self.n_node_ids = data['n_node_ids']
        self.n_conn_nodes = data['n_conn_nodes']
        self.n_labels = config.n_labels
        self.wce_val = data['wce']
        self.oe_id = data['outer_epoch']
        self.max_oe = data['max_oe']

        # Sparse Features
        self.sparse_features = config.sparse_features

        if config.add_labels:
            self.prev_pred = data['labels']

        self.conv_layer = config.kernel_class
        self.n_layers = config.max_depth + 1
        self.skip_conn = config.skip_connections
        self.shared_weights = config.shared_weights

        self.input_dims = config.n_features
        self.output_dims = config.n_labels

        self.sparse_inputs = [self.sparse_features] + [False] * (self.n_layers)
        self.dims = [self.input_dims] + config.dims + [config.dims[-1]] + [config.dims[-1]] + [self.output_dims]
        self.act = [tf.nn.relu] * (self.n_layers)
        self.act.append(lambda x: x)

        self.dropouts = [self.data['dropout_conv']] * (self.n_layers)
        self.drop_fuse = self.data['dropout_out']

        self.optimizer = config.opt(learning_rate=data['lr'])
        self.density = data['batch_density']

        self.feature_names = (config.node_features, config.neighbor_features)

        self.values = []
        self.build()
        self.predictions = self.predict()

    def _build(self):
        # TODO Featureless

        self.layers.append(
            Dense(input_dim=self.dims[0], output_dim=self.dims[1], nnz_features=self.data['nnz_features'],
                  dropout=self.dropouts[0],
                  act=self.act[0], bias=self.bias,
                  sparse_inputs=self.sparse_inputs[0], logging=self.logging, model_name=self.name))

        for i in range(1, self.n_layers):
            self.layers.append(self.conv_layer(layer_id=i, x_names=self.feature_names, dims=self.dims,
                                               dropout=self.dropouts[i], act=self.act[i], bias=self.bias,
                                               shared_weights=self.shared_weights, nnz_features=self.data['nnz_features'],
                                               sparse_inputs=self.sparse_inputs[i], skip_connection=self.skip_conn,
                                               add_labels=self.add_labels, logging=self.logging, model_name=self.name))

        self.layers.append(Fusion(n_layers=self.n_layers-1, x_names=self.feature_names,
                                  input_dim=self.dims[1], output_dim=self.output_dims,
                                  dropout=self.drop_fuse,
                                  act=(lambda x: x), bias=self.bias,
                                  logging=self.logging, model_name=self.name))

    def predict(self):
        predictions = tf.slice(self.outputs, [0, 0], [self.n_node_ids, self.n_labels])
        if self.add_labels:
            prev_predictions = tf.slice(self.prev_pred, [0, 0], [self.n_node_ids, self.n_labels])
            alpha = tf.cast(self.oe_id / self.max_oe, dtype=tf.float32)
            if self.multilabel:
                return (1 - alpha) * tf.nn.sigmoid(predictions) + (alpha) * prev_predictions
            else:
                return (1 - alpha) * tf.nn.softmax(predictions) + (alpha) * prev_predictions
        else:
            if self.multilabel:
                return tf.nn.sigmoid(predictions)
            else:
                return tf.nn.softmax(predictions)

    def _loss(self):
        self.loss = 0

        # Cross entropy Loss
        predictions = tf.slice(self.outputs, [0, 0], [self.n_node_ids, self.n_labels])
        self.ce_loss = sigmoid_binary_cross_entropy(predictions, self.data['targets'], self.wce_val, self.multilabel, self.n_labels)
        self.loss += self.ce_loss

        # L2 Loss
        for v in tf.trainable_variables():
            reject_cands = ['bias']
            # sub_names = v.name.split("/")
            if v.name not in reject_cands:  # and sub_names[2][:7] not in ['weights']:
                self.loss += self.l2 * tf.nn.l2_loss(v)
        self.loss *= self.density
        tf.summary.scalar('loss', self.loss)

    def _accuracy(self):
        self.metric_values = {}
        predictions = self.predict()

        # Get Accuracy
        self.metric_values['mc_accuracy'] = compute_accuracy(predictions, self.data['targets'], multilabel=False)
        self.metric_values['ml_accuracy'] = compute_accuracy(predictions, self.data['targets'], multilabel=True)

        # Get Balanced Absolute error
        self.metric_values['bae'] = get_bae(predictions, self.data['targets'], self.output_dims)

        # Get F1 scores
        if not self.multilabel:
            indices = tf.argmax(predictions, 1)
            predictions = tf.one_hot(indices=indices, on_value=1, off_value=0, depth=self.n_labels)
        else:
            predictions = tf.round(predictions)

        predictions = tf.cast(predictions, dtype=tf.int64)
        truth = tf.cast(self.data['targets'], dtype=tf.int64)
        self.metric_values['micro_f1'] = compute_f1(predictions, truth, self.n_labels, f1_type='micro')
        self.metric_values['macro_f1'] = compute_f1(predictions, truth, self.n_labels, f1_type='macro')
