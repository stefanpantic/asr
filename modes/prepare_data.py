import click


@click.command(name='prepare-data', help='Transform raw dataset into format used for training.')
def prepare_data(**options):
    print('Data preparation function')
