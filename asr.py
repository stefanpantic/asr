import click

from modes.export import export
from modes.infer import infer
from modes.prepare_data import prepare_data
from modes.train import train

if __name__ == '__main__':
    cli = click.Group()
    cli.add_command(export)
    cli.add_command(infer)
    cli.add_command(prepare_data)
    cli.add_command(train)
    cli()
