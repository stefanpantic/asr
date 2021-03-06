import tensorflow as tf


class BatchNormalization1D:
    """Applies batch normalization to both sequence and channel axis in a 1D input."""

    def __init__(self, **kwargs):
        self._norm = tf.keras.layers.BatchNormalization(**kwargs)

    def __call__(self, inputs, training=None):
        expanded = tf.expand_dims(inputs, axis=1)
        normalized = self._norm(expanded, training=training)
        result = tf.squeeze(normalized, axis=1)
        return result
