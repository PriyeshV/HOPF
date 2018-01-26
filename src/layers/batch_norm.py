import tensorflow as tf
from src.layers.layer import Layer


class BatchNorm(Layer):
    """Wrapper for batch normalization implementation in tensorfow"""
    def __init__(self, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)
        pass

    def _call(self, inputs):
        output = tf.contrib.layers.batch_norm(inputs)
        return output
