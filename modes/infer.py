import click


@click.command(name='infer', help='Perform inference.')
@click.option('--model_dir', required=True, help='Path to exported model.')
def infer(**options):
    print('Inference function')
