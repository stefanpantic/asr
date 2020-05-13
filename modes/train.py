import click
import tensorflow as tf
import numpy as np

from models.jasper import Jasper
from models.quartznet import QuartzNet
from utilities.losses import ctc_loss
from utilities.optimizers import NovoGrad
from utilities.wrappers import AutomaticLossScaler, MixedPrecisionOptimizerWrapper


@click.command(name='train', help='Train ASR model.')
@click.option('--dataset', required=True, help='Path to training dataset.')
@click.option('--log_dir', default='./logs/jasper', help='Where to log weights and graphs.')
@click.option('--b', default=1, help='Jasper "B" parameter.')
@click.option('--r', default=5, help='Jasper "R" parameter.')
@click.option('--lr', default=1e-3, help='Model learning rate.')
@click.option('--batch_size', default=16, help='Model batch size.')
@click.option('--epochs', default=400, help='Model batch size.')
def train(**options):
    # TODO: Load dataset
    mfcc = np.ones((1000, 3000, 64), dtype=np.float32)
    seq_lens = np.ones((1000,), dtype=np.int32)
    labels = np.ones((1000, 300), dtype=np.int32)

    # Create dataset from input tensors
    dataset = tf.data.Dataset.from_tensor_slices((mfcc, seq_lens, labels))

    # Prepare datasets
    dataset = dataset.repeat(options['epochs']).batch(options['batch_size'])
    train_it = dataset.make_one_shot_iterator()
    mfcc_inputs, seq_lens_inputs, labels_inputs = train_it.get_next()
    labels_inputs = tf.contrib.layers.dense_to_sparse(labels_inputs)

    quartz = QuartzNet(b=1, r=3)
    logits = quartz(mfcc_inputs)

    # Calculate CTC loss
    ctc = ctc_loss(logits, labels_inputs, seq_lens_inputs)

    # Setup optimizer
    with tf.name_scope('optimizer'):
        g_step = tf.train.get_or_create_global_step()
        lr = tf.train.polynomial_decay(learning_rate=options['lr'],
                                       global_step=g_step,
                                       decay_steps=int(1e7),
                                       end_learning_rate=1e-7,
                                       power=2.0)
        novo_grad = NovoGrad(learning_rate=lr)
        scaler = AutomaticLossScaler(algorithm='backoff')
        optimizer = MixedPrecisionOptimizerWrapper(novo_grad, scaler)
        train_op = optimizer.minimize(ctc)

    # Create session config
    gpu_options = tf.GPUOptions(allow_growth=True)
    tf_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    # Create training hooks
    with tf.train.MonitoredTrainingSession(checkpoint_dir=options['log_dir'],
                                           config=tf_config,
                                           save_summaries_steps=5) as sess:
        # Run training loop
        while not sess.should_stop():
            sess.run([train_op])
