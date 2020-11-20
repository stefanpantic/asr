import glob
import os

import tensorflow as tf


def _generate_feats_and_label_batch(filenames, batch_size, epochs=None, shuffle=False):
    """Construct a queued batch of spectral features and transcriptions.

    Parameters
    ----------
    filenames:
        queue of filenames to read datasets from.
    batch_size:
        Number of utterances per batch.
    epochs:
        Number of epochs, None means infinite.
    shuffle:
        Whether to shuffle input filenames.

    Returns
    -------
    feats:
        spectrograms. 4D tensor of [batch_size, height, width, 3] size.
    labels:
        transcripts. List of length batch_size.
    seq_lens:
        Sequence Lengths. List of length batch_size.
    it:
        Dataset iterator.
    """

    def _parse_example(serialized_example):
        # Define how to parse the example
        context_features = {
            "seq_len": tf.FixedLenFeature([], dtype=tf.int64),
            "labels": tf.VarLenFeature(dtype=tf.int64)
        }
        sequence_features = {
            # Features are 161 dimensional
            "feats": tf.FixedLenSequenceFeature([64, ], dtype=tf.float32)
        }

        # Parse the example (returns a dictionary of tensors)
        ctx_parsed, seq_parsed = tf.parse_single_sequence_example(
            serialized=serialized_example,
            context_features=context_features,
            sequence_features=sequence_features
        )

        return ctx_parsed, seq_parsed

    if shuffle:
        import random
        random.shuffle(filenames)

    # Make tf.datasets.Dataset from filenames
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    dataset = dataset.map(_parse_example)

    if epochs is not None:
        dataset.repeat(epochs)
    else:
        dataset.repeat()

    it = dataset.make_one_shot_iterator()
    context_parsed, sequence_parsed = it.get_next()

    # Generate a batch worth of examples after bucketing
    seq_len, (feats, labels) = tf.contrib.training.bucket_by_sequence_length(
        input_length=tf.cast(context_parsed['seq_len'], tf.int32),
        tensors=[sequence_parsed['feats'], context_parsed['labels']],
        batch_size=batch_size,
        bucket_boundaries=list(range(100, 1900, 100)),
        allow_smaller_final_batch=True,
        num_threads=16,
        dynamic_pad=True)

    return feats, tf.cast(labels, tf.int32), seq_len


def inputs(eval_data, data_dir, batch_size, epochs=None, shuffle=False):
    """Construct input for evaluation using the Reader ops.

    Parameters
    ----------
    eval_data:
        bool, indicating if one should use the train or eval datasets set.
    data_dir:
        Path to the datasets directory.
    batch_size:
        Number of images per batch.
    epochs:
        Number of epochs, None means infinite.
    shuffle:
        bool, whether to shuffle datasets.
    """
    if eval_data == 'train':
        filenames = glob.glob(os.path.join(data_dir, 'train*', '*.tfrecords'))
    elif eval_data == 'val':
        filenames = glob.glob(os.path.join(data_dir, 'dev*', '*.tfrecords'))
    elif eval_data == 'test':
        filenames = glob.glob(os.path.join(data_dir, 'test*', '*.tfrecords'))
    else:
        raise NotImplementedError

    for file in filenames:
        if not os.path.exists(file):
            raise ValueError('Failed to find file: ' + file)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_feats_and_label_batch(filenames, batch_size, epochs, shuffle)


def create_train_inputs(data_dir, batch_size, epochs, shuffle=False):
    """Fetch features, labels and sequence_lengths."""
    with tf.device('/cpu'):
        feats, labels, seq_lens = inputs(eval_data='train',
                                         data_dir=data_dir,
                                         batch_size=batch_size,
                                         epochs=epochs,
                                         shuffle=shuffle)

    return feats, labels, seq_lens


def create_val_inputs(data_dir):
    """Fetch features, labels and sequence_lengths."""
    with tf.device('/cpu'):
        feats, labels, seq_lens = inputs(eval_data='val',
                                         data_dir=data_dir,
                                         batch_size=1,
                                         shuffle=False)

    return feats, labels, seq_lens
