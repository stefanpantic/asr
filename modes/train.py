import click
import tensorflow as tf

from models.jasper import Jasper


@click.command(name='train', help='Train ASR model.')
@click.option('--dataset', required=True, help='Path to training dataset.')
@click.option('--log_dir', default='./logs/jasper', help='Where to log weights and graphs.')
@click.option('--b', default=1, help='Jasper "B" parameter.')
@click.option('--r', default=5, help='Jasper "R" parameter.')
@click.option('--lr', default=1e-3, help='Model learning rate.')
@click.option('--batch_size', default=16, help='Model batch size.')
def train(**options):
    with tf.Session() as sess:
        # TODO: Test model implementation.
        input_ph = tf.placeholder(dtype=tf.float32, shape=[None, None, 64], name='input')
        jasper = Jasper(b=1, r=3)
        logits = jasper(input_ph)
        writer = tf.summary.FileWriter('./logs', sess.graph)
        print(jasper.get_number_of_trainable_variables())

