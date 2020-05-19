import os

import glob2
import numpy as np
import soundfile as sf

from utilities.datasets.utilities import compute_mfcc


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


def process_librispeech_data(partition):
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
                feats[audio_file] = compute_mfcc(audio, sample_rate)
                utt_len[audio_file] = feats[audio_file].shape[0]
                target = ' '.join(parts[1:])
                transcripts[audio_file] = [char_to_ind[i] for i in target]
                print(f"file[{audio_file}] -- "
                      f"utterance length: {utt_len[audio_file]}, "
                      f"transcripts length: {len(transcripts[audio_file])}")
    return feats, transcripts, utt_len


