import os

import click
import tensorflow as tf
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib

from utilities.graph import load_graph


@click.command(name='export', help='Export Tensorflow model to a format used for inference.')
@click.option('--model_dir', required=True, help='Path to serialized model.')
@click.option('--output_dir', default='./export/jasper', help='Where to output converted model.')
def export(**options):
    # Freeze graph
    latest_checkpoint = tf.train.latest_checkpoint(options['model_dir'])
    freeze_graph.freeze_graph(os.path.join(options['model_dir'], 'graph.pbtxt'), "", False, latest_checkpoint,
                              "output", 'save/restore_all', 'save/Const:0',
                              os.path.join(options['model_dir'], 'frozen_graph.pb'), True, '')
    # Optimize graph
    frozen_graph = load_graph(os.path.join(options['model_dir'], 'frozen_graph.pb'), prefix='export')
    optimized_graph = optimize_for_inference_lib.optimize_for_inference(frozen_graph.as_graph_def(),
                                                                        ["export/input"],  # an array of input node(s)
                                                                        ["export/output"],  # an array of output node(s)
                                                                        tf.int32.as_datatype_enum)
    # Serialize optimized graph
    with tf.gfile.FastGFile(os.path.join(options['model_dir'], 'optimized_graph.pb'), 'w') as f:
        f.write(optimized_graph.SerializeToString())
