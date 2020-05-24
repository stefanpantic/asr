import os
from statistics import mean

import click
import tensorflow as tf
from tqdm import tqdm

from models.jasper import Jasper
from models.quartznet import QuartzNet
from utilities.datasets.read import create_train_inputs, create_val_inputs
from utilities.decoders import beam_search_decoder
from utilities.losses import ctc_loss
from utilities.metrics import word_error_rate
from utilities.optimizers import NovoGrad
from utilities.wrappers import AutomaticLossScaler, MixedPrecisionOptimizerWrapper


@click.command(name='train', help='Train ASR model.')
@click.option('--dataset', required=True, help='Path to training dataset.')
@click.option('--model', required=True, default='quartznet', type=click.Choice(['quartznet',
                                                                                'jasper']), help='Which model to use.')
@click.option('--log_dir', default='./logs/train', help='Where to log weights and graphs.')
@click.option('--b', default=1, help='Model "B" parameter.')
@click.option('--r', default=5, help='Model "R" parameter.')
@click.option('--lr', default=1e-3, help='Model learning rate.')
@click.option('--batch_size', default=16, help='Model batch size.')
@click.option('--epochs', default=400, help='Model batch size.')
@click.option('--calculate_val_summaries_steps', default=1000, help='After how many steps to calculate metrics.')
def train(**options):
    if not os.path.exists(options['dataset']):
        raise ValueError(f'Invalid option for --dataset, directory: {options["dataset"]} doesn\'t exist.')

    # Prepare train dataset
    mfcc_train_ins, labels_train_ins, seq_lens_train_ins, _ = create_train_inputs(data_dir=options['dataset'],
                                                                                  batch_size=options['batch_size'],
                                                                                  epochs=options['epochs'],
                                                                                  shuffle=True)
    # Prepare validation dataset
    mfcc_val_ins, labels_val_ins, seq_lens_val_ins, val_it = create_val_inputs(data_dir=options['dataset'])

    # Model class catalogue
    models = {
        'quartznet': QuartzNet,
        'jasper': Jasper,
    }

    try:
        model_cls = models[options['model']]
    except KeyError:
        raise NotImplementedError(f'Invalid model {options["model"]}')

    # Construct model
    model = model_cls(b=options['b'], r=options['r'])

    # Get train outputs
    train_logits = model(mfcc_train_ins)
    train_decoded = beam_search_decoder(train_logits, seq_lens_train_ins, beam_width=5)
    train_wer = word_error_rate(train_decoded, labels_train_ins)

    # Get validation outputs
    val_logits = model(mfcc_val_ins)
    val_decoded = beam_search_decoder(val_logits, seq_lens_val_ins, beam_width=5)
    val_wer = word_error_rate(val_decoded, labels_val_ins)

    # Calculate CTC loss
    prep_conv_size = model.get_layer_configuration()['prep_config']['kernel_size']
    train_ctc = ctc_loss(train_logits, labels_train_ins, seq_lens_train_ins, prep_conv_size)
    val_ctc = ctc_loss(val_logits, labels_val_ins, seq_lens_val_ins, prep_conv_size)

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
        train_op = optimizer.minimize(train_ctc)

    # Create summary writer and operations
    writer = tf.summary.FileWriter(logdir=options['log_dir'])

    # Create summaries fn
    def _create_summaries(name):
        ctc_ph = tf.placeholder(dtype=tf.float32, shape=[], name=f'{name}_ctc')
        wer_ph = tf.placeholder(dtype=tf.float32, shape=[], name=f'{name}_wer')
        ctc_op = tf.summary.scalar(name=f'{name}/ctc_loss')
        wer_op = tf.summary.scalar(name=f'{name}/wer')
        writer.add_summary(ctc_op)
        writer.add_summary(wer_op)

        return ctc_op, wer_op, ctc_ph, wer_ph

    # Create summaries
    train_ctc_op, train_wer_op, train_ctc_ph, train_wer_ph = _create_summaries('train')
    val_ctc_op, val_wer_op, val_ctc_ph, val_wer_ph = _create_summaries('val')

    # Create session config
    gpu_options = tf.GPUOptions(allow_growth=True)
    tf_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    class _DatasetInitializerHook(tf.train.SessionRunHook):
        def __init__(self, *args):
            self._args = args

        def begin(self):
            pass

        def after_create_session(self, s, _):
            s.run(self._args)

    # Create training hooks
    init_hook = _DatasetInitializerHook(val_it.initializer)
    with tf.train.MonitoredTrainingSession(checkpoint_dir=options['log_dir'],
                                           config=tf_config,
                                           hooks=[init_hook],
                                           summary_dir=None,
                                           save_summaries_secs=None,
                                           save_summaries_steps=None) as sess:
        # Run training loop
        while not sess.should_stop():
            # Train
            for i in tqdm(range(options['calculate_val_summaries_steps'])):
                sess.run([train_op])

                # Calculate train metrics
                if i % 10 == 0:
                    item_loss, item_wer = sess.run([train_ctc, train_wer])
                    sess.run(train_ctc_op, feed_dict={train_ctc_ph: item_loss})
                    sess.run(train_wer_op, feed_dict={train_wer_ph: item_wer})

            # Calculate validation metrics
            val_losses = []
            val_wers = []
            while True:
                try:
                    item_loss, item_wer = sess.run([val_ctc, val_wer])
                    val_losses.append(item_loss)
                    val_wers.append(item_wer)
                except tf.errors.OutOfRangeError:
                    sess.run(val_it.initializer)
                    sess.run(val_ctc_op, feed_dict={val_ctc_ph: mean(val_losses)})
                    sess.run(val_wer_op, feed_dict={val_wer_ph: mean(val_losses)})
                    break
