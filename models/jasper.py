import tensorflow as tf

from models.base import AbstractModelBase


class Jasper(AbstractModelBase):
    """NVIDIA's Jasper model from https://arxiv.org/pdf/1904.03288.pdf"""

    def __init__(self, b, r, alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ\' '):
        super().__init__(b=b, r=r, alphabet=alphabet, conv_cls=tf.keras.layers.Conv1D, name='Jasper')

    def get_layer_configuration(self):
        """
        >>> AbstractModelBase.get_layer_configuration
        """
        layer_configuration = {
            'prep_config': {
                'filters': 256,
                'kernel_size': 11,
                'dropout_rate': 0.2,
            },
            'block_config': {
                'filters': [256, 384, 512, 640, 768],
                'kernel_size': [11, 13, 17, 21, 25],
                'dropout_rate': [0.2, 0.2, 0.2, 0.3, 0.3],
            },
            'post_config': {
                'filters': [896, 1024],
                'kernel_size': [29, 1],
                'dilation_rate': [2, 1],
                'dropout_rate': [0.4, 0.4],
            },
        }

        return layer_configuration
