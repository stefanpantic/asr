import tensorflow as tf


def load_graph(model_file, prefix=None):
    """Load Tensorflow graph.

    Parameters
    ----------
    model_file:
        Path to graph.pbtxt
    prefix:
        Prefix to append to node names.

    Returns
    -------
        graph: Parsed Tensorflow graph
    """
    graph_def = tf.GraphDef()
    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name=prefix)

    return graph
