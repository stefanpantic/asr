import tensorflow as tf


def ctc_loss(logits, labels, seq_lens, prep_conv_kernel_size):
    """Computes Connectionist Temporal Classification loss.

    Parameters
    ----------
    logits:
        Unscaled network outputs.
    labels:
        True labels.
    seq_lens
        Input sequence lengths.
    prep_conv_kernel_size:
        Kernel size of preprocess conv layer. Used to compute sequence_length.

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
                         inputs=tf.transpose(logits, [1, 0, 2]),
                         sequence_length=tf.cast(tf.floor((seq_lens - prep_conv_kernel_size) / 2) + 1, tf.int32))

    total_loss = mask_nans(ctc)
    loss = tf.reduce_mean(total_loss)
    return loss
