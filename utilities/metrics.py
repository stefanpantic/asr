import tensorflow as tf


def word_error_rate(predictions, labels):
    """Calculate WER metric.

    Parameters
    ----------
    predictions:
        Predicted values.
    labels
        Actual values.
    Returns
    -------
        Calculated WER.
    """
    distance = tf.reduce_sum(tf.edit_distance(predictions, labels, normalize=False))
    reference_length = tf.cast(tf.size(labels, out_type=tf.int32), dtype=tf.float32)
    return distance / reference_length


def label_error_rate(predictions, labels):
    """Calculate LER metric.

    Parameters
    ----------
    predictions:
        Predicted values.
    labels
        Actual values.
    Returns
    -------
        Calculated LER.
    """
    return tf.reduce_mean(tf.edit_distance(predictions, labels, normalize=False))
