import tensorflow as tf


class BatchNormalization1D:
    """Applies batch normalization to both sequence and channel axes in a 1D input."""

    def __init__(self, **kwargs):
        self._norm = tf.keras.layers.BatchNormalization(**kwargs)

    def call(self, inputs, training=None):
        expanded = tf.expand_dims(inputs, axis=1)
        result = self._norm(expanded, training=training)
        return result
