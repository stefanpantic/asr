import tensorflow as tf


def beam_search_decoder(logits, seq_lens, beam_width=10, to_dense=False):
    """Decodes model outputs using beam search

    Parameters
    ----------
    logits:
        Unscaled model outputs.
    seq_lens:
        Input sequence lengths.
    beam_width:
        Beam search beam width.
    to_dense:
        Whether to convert decoded indices to a dense tensor.

    Returns
    -------
        decoded: Decoded model outputs.
    """
    decoder_inputs = tf.transpose(logits, [1, 0, 2])
    outputs, _ = tf.nn.ctc_beam_search_decoder_v2(decoder_inputs,
                                                  sequence_length=seq_lens,
                                                  beam_width=beam_width,
                                                  top_paths=1)

    decoded_outputs = tf.sparse.to_dense(outputs[0]) if to_dense else outputs[0]
    return tf.cast(decoded_outputs, tf.int32)
