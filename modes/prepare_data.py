import os

import click
import wget

from utilities.data.preprocess import create_records


@click.command(name='prepare-data', help='Transform raw dataset into format used for training.')
@click.option('--download_path', default=None, help='Where to download dataset from (if specified).')
@click.option('--dataset', required=True, type=click.Choice(['librispeech', 'common_voice']), help='Which dataset.')
@click.option('--dataset_path', required=True, help='Path to training dataset.')
@click.option('--output_path', required=True, help='Where to store .tfrecords.')
def prepare_data(**options):
    if options['download_path'] is not None:
        if not os.path.exists(options['dataset_path']):
            os.makedirs(options['dataset_path'], exist_ok=False)

        wget.download(options['download_path'], out=options['dataset_path'])

        import tarfile
        _, file = options['download_path'].rsplit('/', 1)
        tr = tarfile.open(os.path.join(options['dataset_path'], file))
        tr.extractall(options['dataset_path'])

    if options['dataset'] == 'librispeech':
        create_records(os.path.join(options['dataset_path'], 'LibriSpeech'), options['output_path'])
    else:
        raise NotImplementedError
