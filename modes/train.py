import os

import click
import tensorflow as tf
import numpy as np

from models.jasper import Jasper
from models.quartznet import QuartzNet
from utilities.datasets.read import create_train_inputs, create_val_inputs
from utilities.losses import ctc_loss
from utilities.optimizers import NovoGrad
from utilities.wrappers import AutomaticLossScaler, MixedPrecisionOptimizerWrapper


@click.command(name='train', help='Train ASR model.')
@click.option('--dataset', required=True, help='Path to training dataset.')
@click.option('--log_dir', default='./logs/train', help='Where to log weights and graphs.')
@click.option('--b', default=1, help='Model "B" parameter.')
@click.option('--r', default=5, help='Model "R" parameter.')
@click.option('--lr', default=1e-3, help='Model learning rate.')
@click.option('--batch_size', default=16, help='Model batch size.')
@click.option('--epochs', default=400, help='Model batch size.')
def train(**options):
    if not os.path.exists(options['dataset']):
        raise ValueError(f'Invalid option for --dataset, directory: {options["dataset"]} doesn\'t exist.')

    # Prepare train dataset
    mfcc_train_ins, labels_train_ins, seq_lens_train_ins = create_train_inputs(data_dir=options['dataset'],
                                                                               batch_size=options['batch_size'],
                                                                               epochs=options['epochs'], shuffle=True)
    # Prepare validation dataset
    mfcc_val_ins, labels_val_ins, seq_lens_val_ins = create_val_inputs(data_dir=options['dataset'])

    quartz = QuartzNet(b=1, r=3)
    train_logits = quartz(mfcc_train_ins)

    # TODO: Calculate and log validation metrics
    val_logits = quartz(mfcc_val_ins)

    # Calculate CTC loss
    ctc = ctc_loss(train_logits, labels_train_ins, seq_lens_train_ins)

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
                                           summary_dir=None) as sess:
        count = 0
        # Run training loop
        while not sess.should_stop():
            # TODO: Calculate and log train metrics
            sess.run([train_op])
            count += 1
            if count % 100 == 0:
                print(f'Executed {count} batches...')

