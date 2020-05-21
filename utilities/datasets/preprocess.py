import os
import shutil

import glob2
import tensorflow as tf
from tqdm import tqdm

from utilities.datasets.commonvoice import process_common_voice_data
from utilities.datasets.librispeech import process_librispeech_data


def make_example(seq_len, spec_feat, labels):
    """ Creates a SequenceExample for a single utterance.
        This function makes a SequenceExample given the sequence length,
        mfcc features and corresponding transcript.
        These sequence examples are read using tf.parse_single_sequence_example
        during training.
        Note: Some of the tf modules used in this function(such as
        tf.train.Feature) do not have comprehensive documentation in v0.12.
        This function was put together using the test routines in the
        tensorflow repo.

    Parameters
    ----------
        seq_len: integer represents the sequence length in time frames.
        spec_feat: [TxF] matrix of mfcc features.
        labels: list of ints representing the encoded transcript.

    Returns
    -------
        Serialized sequence example.
    """
    # Feature lists for the sequential features of the example
    feats_list = [tf.train.Feature(float_list=tf.train.FloatList(value=frame))
                  for frame in spec_feat]
    feat_dict = {"feats": tf.train.FeatureList(feature=feats_list)}
    sequence_feats = tf.train.FeatureLists(feature_list=feat_dict)

    # Context features for the entire sequence
    len_feat = tf.train.Feature(int64_list=tf.train.Int64List(value=[seq_len]))
    label_feat = tf.train.Feature(int64_list=tf.train.Int64List(value=labels))

    context_feats = tf.train.Features(feature={"seq_len": len_feat,
                                               "labels": label_feat})

    ex = tf.train.SequenceExample(context=context_feats,
                                  feature_lists=sequence_feats)

    return ex.SerializeToString()


def create_records(audio_path, output_path, dataset):
    """ Pre-processes the raw audio and generates TFRecords.
        This function computes the mfcc features, encodes string transcripts
        into integers, and generates sequence examples for each utterance.
        Multiple sequence records are then written into TFRecord files.

    Parameters
    ----------
    audio_path:
        Path to dataset.
    output_path:
        Where to write .tfrecords.
    dataset:
        Either 'librispeech' or 'commonvoice'. Determines which dataset format to parse.
    """
    assert os.path.exists(audio_path), f'Invalid audio path: {audio_path}. Path doesn\'t exist.'
    assert dataset.lower() in ['librispeech', 'commonvoice'], f'Invalid dataset parameter: {dataset}. ' \
                                                              f'Must be one of "librispeech", "commonvoice"'
    dataset = dataset.lower()

    for partition in sorted(glob2.glob(os.path.join(audio_path, '*'))):
        if dataset == 'librispeech':
            if os.path.isfile(partition):
                continue

            print('Processing ' + partition)
            feats, transcripts, utt_len = process_librispeech_data(partition)
            write_suffix = partition.split(os.path.sep)[-1]
        elif dataset == 'commonvoice':
            if os.path.isdir(partition) or any(e in partition for e in ['invalidated', 'other']):
                continue

            print('Processing ' + partition)
            feats, transcripts, utt_len = process_common_voice_data(partition)
            write_suffix, _ = os.path.splitext(os.path.basename(partition))
        else:
            raise NotImplementedError

        sorted_utts = sorted(utt_len, key=utt_len.get)

        # bin into groups of 100 frames.
        max_t = int(utt_len[sorted_utts[-1]] / 100)
        min_t = int(utt_len[sorted_utts[0]] / 100)

        # Create destination directory
        write_dir = os.path.join(output_path, write_suffix)
        if os.path.exists(write_dir):
            shutil.rmtree(write_dir)
        os.makedirs(write_dir)

        if 'train' in os.path.basename(partition):
            # Create multiple TFRecords based on utterance length for training
            writer = {}
            count = {}
            print('Processing training files...')
            for i in range(min_t, max_t + 1):
                filename = os.path.join(write_dir, 'train' + '_' + str(i) + '.tfrecords')
                writer[i] = tf.io.TFRecordWriter(filename)
                count[i] = 0

            for utt in tqdm(sorted_utts, desc='Writing TFRecords'):
                example = make_example(utt_len[utt], feats[utt].tolist(), transcripts[utt])
                index = int(utt_len[utt] / 100)
                writer[index].write(example)
                count[index] += 1

            for i in range(min_t, max_t + 1):
                writer[i].close()
            print(count)

            # Remove bins which have fewer than 20 utterances
            for i in range(min_t, max_t + 1):
                if count[i] < 20:
                    os.remove(os.path.join(write_dir, 'train' + '_' + str(i) + '.tfrecords'))
        else:
            # Create single TFRecord for dev and test partition
            filename = os.path.join(write_dir, os.path.basename(write_dir) + '.tfrecords')
            print('Creating', filename)
            record_writer = tf.io.TFRecordWriter(filename)
            for utt in tqdm(sorted_utts, desc='Writing TFRecords'):
                example = make_example(utt_len[utt], feats[utt].tolist(), transcripts[utt])
                record_writer.write(example)
            record_writer.close()
            print('Processed ' + str(len(sorted_utts)) + ' audio files')
