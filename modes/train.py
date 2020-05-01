import click


@click.command(name='train', help='Train ASR model.')
@click.option('--dataset', required=True, help='Path to training dataset.')
@click.option('--log_dir', default='./logs/jasper', help='Where to log weights and graphs.')
@click.option('--b', default=1, help='Jasper "B" parameter.')
@click.option('--r', default=5, help='Jasper "R" parameter.')
@click.option('--lr', default=1e-3, help='Model learning rate.')
@click.option('--batch_size', default=16, help='Model batch size.')
def train(**options):
    print('Training function')
