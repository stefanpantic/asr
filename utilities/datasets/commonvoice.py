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
    paths_to_labels = {k: v for k, v in tmp.items() if all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ\' ' for c in v)}
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
    # NOTE: This is kind of hacky but I don't see a workaround.
    import os
    import sys
    import pathlib
    import tempfile
    import random
    import string

    def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
        return ''.join(random.choice(chars) for _ in range(size))

    ap = os.path.abspath(__file__)
    dn = os.path.dirname(ap)
    pth = pathlib.Path(dn)
    platform = 'linux' if sys.platform.startswith('linux') else os.path.join('win32', 'bin')
    os.environ['PATH'] += os.pathsep + os.path.join(pth.parent.parent, 'external', 'ffmpeg', platform)

    import pydub

    mp3_pth_to_lab = read_and_parse_tsv(tsv_path)
    features = {}
    data_dir = os.path.join(os.path.abspath(os.path.dirname(tsv_path)), 'clips')
    with tempfile.TemporaryDirectory() as tempdir:
        for file_name in mp3_pth_to_lab.keys():
            mp3_path = os.path.join(data_dir, file_name)
            mp3_data = pydub.AudioSegment.from_mp3(mp3_path)

            flac_name = f'{id_generator()}.flac'
            flac_path = os.path.join(tempdir, flac_name)

            mp3_data.export(flac_path, format='flac', parameters=['-ar', '16000'])
            audio, sr = sf.read(flac_path)
            features[file_name] = compute_mfcc(audio, sr)

            os.remove(flac_path)

    transcripts = mp3_pth_to_lab
    utt_len = {pth: feat.shape[0] for pth, feat in features.items()}

    return features, transcripts, utt_len
