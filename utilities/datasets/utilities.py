import numpy as np
from python_speech_features import mfcc


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

    mfcc_feat = mfcc(audio_data, sample_rate, winlen=0.025, winstep=0.01,
                     numcep=64, nfilt=64, nfft=512, lowfreq=0, highfreq=None,
                     preemph=0.97, ceplifter=22, appendEnergy=True)

    mfcc_feat = mfcc_feat - np.mean(mfcc_feat)
    mfcc_feat = mfcc_feat / (np.std(mfcc_feat) + 1e-8)

    return mfcc_feat
