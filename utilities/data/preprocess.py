import os

import glob2
import numpy as np
import soundfile as sf
import tensorflow as tf
from python_speech_features import mfcc
from tqdm import tqdm


def compute_linear_spectrogram(samples,
                               sample_rate,
                               stride_ms=10.0,
                               window_ms=20.0,
                               max_freq=None,
                               eps=1e-14):
    """Compute the linear spectrogram from FFT energy."""
    if max_freq is None:
        max_freq = sample_rate / 2
    if max_freq > sample_rate / 2:
        raise ValueError("max_freq must not be greater than half of sample rate.")
    if stride_ms > window_ms:
        raise ValueError("Stride size must not be greater than window size.")

    stride_size = int(0.001 * sample_rate * stride_ms)
    window_size = int(0.001 * sample_rate * window_ms)

    spectrogram, freqs = _spectrogram_real(samples,
                                           window_size=window_size,
                                           stride_size=stride_size,
                                           sample_rate=sample_rate)

    ind = np.where(freqs <= max_freq)[0][-1] + 1
    spectrogram = np.log(spectrogram[:ind, :] + eps)

    spectrogram = spectrogram.transpose()

    # z-score normalizer
    spectrogram = spectrogram - np.mean(spectrogram)
    spectrogram = spectrogram / np.std(spectrogram)

    return spectrogram


def _spectrogram_real(samples, window_size, stride_size, sample_rate):
    """Compute the spectrogram for samples from a real signal."""
    # Extract strided windows
    truncate_size = (len(samples) - window_size) % stride_size
    samples = samples[:len(samples) - truncate_size]
    nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
    nstrides = (samples.strides[0], samples.strides[0] * stride_size)
    windows = np.lib.stride_tricks.as_strided(samples, shape=nshape, strides=nstrides)
    assert np.all(windows[:, 1] == samples[stride_size:(stride_size + window_size)])
    # window weighting, squared Fast Fourier Transform (fft), scaling
    weighting = np.hanning(window_size)[:, None]
    fft = np.fft.rfft(windows * weighting, axis=0)
    fft = np.absolute(fft)
    fft = fft ** 2
    scale = np.sum(weighting ** 2) * sample_rate
    fft[1:-1, :] *= (2.0 / scale)
    fft[(0, -1), :] /= scale
    # prepare fft frequency list
    freqs = float(sample_rate) / window_size * np.arange(fft.shape[0])
    return fft, freqs


def compute_mfcc(audio_data, sample_rate):
    """ Computes the mel-frequency cepstral coefficients.
        The audio time series is normalised and its mfcc features are computed.

    Parameters
    ----------
        audio_data: time series of the speech utterance.
        sample_rate: sampling rate.

    Returns
    -------
        mfcc_feat:[num_frames x F] matrix representing the mfcc.
    """

    # z-score normalizer
    audio_data = audio_data - np.mean(audio_data)
    audio_data = audio_data / np.std(audio_data)

    mfcc_feat = mfcc(audio_data, sample_rate, winlen=0.02, winstep=0.01,
                     numcep=13, nfilt=26, nfft=512, lowfreq=0, highfreq=None,
                     preemph=0.97, ceplifter=22, appendEnergy=True)

    return mfcc_feat


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


def process_data(partition):
    """ Reads audio waveform and transcripts from a dataset partition
        and generates mfcc features.

    Parameters
    ----------
        partition - represents the dataset partition name.

    Returns
    -------
        feats: dict containing mfcc feature per utterance
        transcripts: dict of lists representing transcript.
        utt_len: dict of ints holding sequence length of each
                 utterance in time frames.
    """
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ' "
    char_to_ind = {ch: i for (i, ch) in enumerate(alphabet)}

    feats = {}
    transcripts = {}
    utt_len = {}  # Required for sorting the utterances based on length

    for filename in glob2.iglob(partition + '/**/*.txt'):
        with open(filename, 'r') as f:
            for line in f:
                parts = line.split()
                audio_file = parts[0]
                file_path = os.path.join(os.path.dirname(filename), audio_file + '.flac')
                audio, sample_rate = sf.read(file_path)
                # feats[audio_file] = compute_mfcc(audio, sample_rate)
                feats[audio_file] = compute_linear_spectrogram(audio, sample_rate)
                utt_len[audio_file] = feats[audio_file].shape[0]
                target = ' '.join(parts[1:])
                transcripts[audio_file] = [char_to_ind[i] for i in target]
                if ((utt_len[audio_file] - 19) // 2 - 9) // 2 == 60:
                    print("file[%s] -- utterance length: %d, transcripts length: %d" % (
                        audio_file, ((utt_len[audio_file] - 19) // 2 - 9) // 2, len(transcripts[audio_file])))
    return feats, transcripts, utt_len


def create_records(audio_path, output_path):
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
    """
    for partition in sorted(glob2.glob(audio_path + '/*')):
        if os.path.isfile(partition):
            continue
        print('Processing ' + partition)
        feats, transcripts, utt_len = process_data(partition)
        sorted_utts = sorted(utt_len, key=utt_len.get)

        # bin into groups of 100 frames.
        max_t = int(utt_len[sorted_utts[-1]] / 100)
        min_t = int(utt_len[sorted_utts[0]] / 100)

        # Create destination directory
        write_dir = os.path.join(output_path, partition.split('/')[-1])
        if tf.gfile.Exists(write_dir):
            tf.gfile.DeleteRecursively(write_dir)
        tf.gfile.MakeDirs(write_dir)

        if 'train' in os.path.basename(partition):
            # Create multiple TFRecords based on utterance length for training
            writer = {}
            count = {}
            print('Processing training files...')
            for i in range(min_t, max_t + 1):
                filename = os.path.join(write_dir, 'train' + '_' + str(i) + '.tfrecords')
                writer[i] = tf.python_io.TFRecordWriter(filename)
                count[i] = 0

            for utt in tqdm(sorted_utts):
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
            record_writer = tf.python_io.TFRecordWriter(filename)
            for utt in sorted_utts:
                example = make_example(utt_len[utt], feats[utt].tolist(), transcripts[utt])
                record_writer.write(example)
            record_writer.close()
            print('Processed ' + str(len(sorted_utts)) + ' audio files')
