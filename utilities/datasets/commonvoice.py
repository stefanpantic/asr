import re

import pandas as pd
import soundfile as sf

from utilities.datasets.utilities import compute_mfcc


def read_and_parse_tsv(path):
    """Reads and parses a CommonVoice .tsv file.

    Parameters
    ----------
    path:
        Path to .tsv input file.

    Returns
    -------
        paths_to_labels: dict containing a path to audio file as key and sentence as value.
    """
    df = pd.read_csv(path, sep='\t')
    rgx = re.compile(r'[,.!?\\;]')
    labels = [rgx.sub('', lab.upper()) for lab in df.sentence]
    tmp = {pth: lab for pth, lab in zip(df.path, labels)}
    paths_to_labels = {k: v for k, v in tmp if all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ\' ' for c in v)}
    return paths_to_labels


def process_common_voice_data(tsv_path):
    """Process .tsv file located at tsv_path

    Parameters
    ----------
    tsv_path:
        Path to .tsv file with file paths and transcripts.

    Returns
    -------
    feats:
        MFCC features for each sentence.
    transcripts:
        Labels.
    utt_len:
        Sequence length for each label.
    """
    pth_to_lab = read_and_parse_tsv(tsv_path)
    # TODO: Convert mp3 to flac
    features = {path: compute_mfcc(*sf.read(path)) for path in pth_to_lab.keys()}
    transcripts = pth_to_lab
    utt_len = {pth: feat.shape[0] for pth, feat in features.items()}

    return features, transcripts, utt_len
