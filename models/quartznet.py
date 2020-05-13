import tensorflow as tf

from models.base import AbstractModelBase


class QuartzNet(AbstractModelBase):
    """NVIDIA's QuartzNet model from https://arxiv.org/pdf/1910.10261.pdf"""

    def __init__(self, b, r, alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ\''):
        super().__init__(b=b, r=r, alphabet=alphabet, conv_cls=tf.keras.layers.SeparableConv1D, name='QuartzNet')

    def get_layer_configuration(self):
        """
        >>> AbstractModelBase.get_layer_configuration
        """
        layer_configuration = {
            'prep_config': {
                'filters': 256,
                'kernel_size': 33,
                'dropout_rate': 0,
            },
            'block_config': {
                'filters': [256, 256, 512, 512, 512],
                'kernel_size': [33, 39, 51, 63, 75],
                'dropout_rate': [0, 0, 0, 0, 0],
            },
            'post_config': {
                'filters': [87, 1],
                'kernel_size': [512, 1024],
                'dilation_rate': [1, 1],
                'dropout_rate': [0, 0],
            },
        }

        return layer_configuration
