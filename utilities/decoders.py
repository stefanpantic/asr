import tensorflow as tf


def beam_search_decoder(logits, seq_lens, prep_conv_kernel_size, to_dense=False):
    """Decodes model outputs using beam search

    Parameters
    ----------
    logits:
        Unscaled model outputs.
    seq_lens:
        Input sequence lengths.
    prep_conv_kernel_size:
        Kernel size of preprocess conv layer. Used to compute sequence_length.
    to_dense:
        Whether to convert decoded indices to a dense tensor.


    Returns
    -------
        decoded: Decoded model outputs.
    """
    decoder_inputs = tf.transpose(logits, [1, 0, 2])
    input_seq_lens = tf.cast(tf.floor((seq_lens - prep_conv_kernel_size) / 2) + 1, tf.int32)
    outputs, _ = tf.nn.ctc_beam_search_decoder(decoder_inputs,
                                               sequence_length=input_seq_lens,
                                               beam_width=5,
                                               top_paths=1)

    decoded_outputs = tf.sparse.to_dense(outputs[0]) if to_dense else outputs[0]
    return tf.cast(decoded_outputs, tf.int32)
