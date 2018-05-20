import tensorflow as tf


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'wce', 'multilabel'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.wce = kwargs.get('wce')
        self.multilabel = kwargs.get('multilabel')

        self.vars = {}
        self.placeholders = {}

        self.layers = []

        self.data = {}
        self.data['activations'] = []
        self.inputs = None
        self.outputs = None
        self.num_features_nonzero = 0

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None
        self.scores = None

        self.n_layers = None
        self.shared_weights = True
        self.act = None

        self.skip_conn = False
        self.feature_names = ''

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.data['activations'].append(self.inputs)
        for i, layer in enumerate(self.layers):
            print('Layer ', i, ' : ', layer)
            if i == 0 and (self.name not in ['krylov', 'cheby']):
                hidden, h0 = layer(self.data)
                self.data['activations'].append(self.act[i](h0))
            else:
                hidden = layer(self.data)

            # Add skip connections and pass it through and activation layer
            if i != self.n_layers:
                if self.skip_conn and self.name not in ['krylov', 'cheby']:
                    if i != 0 or self.name == 'fusion':  # or True
                        print('Hop Skip connection| From: ', i, ' To: ', i+1, layer)
                        hidden += self.data['activations'][-1]
                hidden = self.act[i](hidden)

            self.data['activations'].append(hidden)
        self.outputs = self.data['activations'][-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Set loss and gradient clipping
        self._loss()
        self._accuracy()

        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var)
                                  for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)
        # self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)
