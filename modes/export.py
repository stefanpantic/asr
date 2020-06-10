import os

import click
import tensorflow as tf
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.keras import backend as K

from models.jasper import Jasper
from models.quartznet import QuartzNet
from utilities.decoders import beam_search_decoder


@click.command(name='export', help='Export Tensorflow model to a format used for inference.')
@click.option('--model_dir', required=True, help='Path to serialized model.')
@click.option('--output_dir', default=None, help='Where to output converted model.')
def export(**options):
    # Model class catalogue
    models = {
        'quartznet': QuartzNet,
        'jasper': Jasper,
    }

    # Construct model
    K.set_learning_phase(0)
    input_ph = tf.placeholder(dtype=tf.float32, shape=[None, 64], name='input')
    expanded_ph = tf.expand_dims(input_ph, axis=0)
    model = QuartzNet(b=1, r=5)
    logits = model(expanded_ph, training=False)
    prep_conv_size = model.get_layer_configuration()['prep_config']['kernel_size']
    outputs = beam_search_decoder(logits, tf.expand_dims(tf.shape(expanded_ph)[1], axis=0),
                                  prep_conv_kernel_size=prep_conv_size, to_dense=True)
    _ = tf.identity(outputs, name='output')

    # Restore model weights
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(options['model_dir']))

    # Convert variables to constants
    constant_graph = tf.graph_util.convert_variables_to_constants(sess,
                                                                  sess.graph_def,
                                                                  ['output'],
                                                                  ['input', 'output'])
    # Serialize optimized graph
    output_dir = options['output_dir'] or options['model_dir']
    with tf.gfile.FastGFile(os.path.join(output_dir, 'frozen_graph.pb'), 'w') as f:
        f.write(constant_graph.SerializeToString())
