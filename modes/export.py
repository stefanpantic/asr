import click


@click.command(name='export', help='Export Tensorflow model to a format used for inference.')
@click.option('--model_dir', required=True, help='Path to serialized model.')
@click.option('--output_dir', default='./export/jasper', help='Where to output converted model.')
def export(**options):
    print('Export function')
