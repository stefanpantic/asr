import tensorflow as tf


def ctc_loss(logits, labels, seq_lens):
    """Computes Connectionist Temporal Classification loss.

    Parameters
    ----------
    logits:
        Unscaled network outputs.
    labels:
        True labels.
    seq_lens
        Input sequence lengths.

    Returns
    -------
        loss: Calculated CTC loss.
    """

    def mask_nans(x):
        x_zeros = tf.zeros_like(x)
        x_mask = tf.is_finite(x)
        y = tf.where(x_mask, x, x_zeros)
        return y

    ctc = tf.nn.ctc_loss(labels=labels,
                         inputs=logits,
                         sequence_length=tf.cast(tf.floor((seq_lens - 33) / 2) + 1, tf.int32),
                         time_major=False)

    total_loss = mask_nans(ctc)
    loss = tf.reduce_mean(total_loss)
    return loss
