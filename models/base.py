from abc import abstractmethod
from functools import reduce

import tensorflow as tf

from layers.batch_normalization import BatchNormalization1D


class AbstractModelBase:
    """Abstracted base model for the NVIDIA Jasper family of models."""

    def __init__(self, b, r, alphabet, conv_cls, name):
        self._b = b
        self._r = r
        self._alphabet = alphabet + '*'  # Add 'blank' symbol
        self._name = name
        self._conv_cls = conv_cls

        # Get layer configurations
        layer_configuration = self.get_layer_configuration()
        prep_config = layer_configuration['prep_config']
        block_config = layer_configuration['block_config']
        post_config = layer_configuration['post_config']

        # Preprocess block layers
        prep_layers = self._build_block(filters=prep_config['filters'],
                                        kernel_size=prep_config['kernel_size'],
                                        strides=2,
                                        dilation_rate=1,
                                        dropout_rate=prep_config['dropout_rate'],
                                        conv_cls=self._conv_cls)[0]
        self._prep_conv, self._prep_norm, self._prep_relu, self._prep_drop = prep_layers

        # Block parameters
        self._block_filters = block_config['filters']
        self._block_kernels = block_config['kernel_size']
        self._block_dropouts = block_config['dropout_rate']
        self._blocks = [self._build_block_with_residual(filters=f, kernel_size=k, dropout_rate=d)
                        for f, k, d in zip(self._block_filters, self._block_kernels, self._block_dropouts)]

        # Postprocess block layers
        post_blocks = [
            self._build_block(filters=f,
                              kernel_size=k,
                              strides=1,
                              dilation_rate=di,
                              dropout_rate=dr,
                              conv_cls=self._conv_cls) for f, k, di, dr in zip(post_config['filters'],
                                                                               post_config['kernel_size'],
                                                                               post_config['dilation_rate'],
                                                                               post_config['dropout_rate'])
        ]
        post_a, post_b = post_blocks
        self._post_blocks = [post_a[0], post_b[0]]

        # Last layer (alphabet + 'blank')
        self._last_layer = tf.keras.layers.Conv1D(filters=len(self._alphabet), kernel_size=1, name='last_layer')

    def __call__(self, inputs, training=None):
        with tf.name_scope(self._name):
            with tf.name_scope('preprocess'):
                prep_conv = self._prep_conv(inputs)
                prep_norm = self._prep_norm(prep_conv, training=training)
                prep_relu = self._prep_relu(prep_norm)
                prep_out = self._prep_drop(prep_relu, training=training)

            with tf.name_scope(f'{self._b}x{self._r}'):
                bxr_out = prep_out
                for b, (layers, res_layers) in enumerate(self._blocks):
                    with tf.name_scope(f'b{b}'):
                        res_conv, res_norm = res_layers

                        r_out = bxr_out
                        for r, (conv, norm, relu, drop) in enumerate(layers):
                            with tf.name_scope(f'b{b}xr{r}'):
                                conv_out = conv(bxr_out)
                                norm_out = norm(conv_out, training=training)

                                if r == self._r - 1:
                                    with tf.name_scope(f'b{b}xr{r}-residual'):
                                        res_conv_out = res_conv(r_out)
                                        res_norm_out = res_norm(res_conv_out, training=training)
                                        norm_out += res_norm_out

                                relu_out = relu(norm_out)
                                bxr_out = drop(relu_out)

            with tf.name_scope('postprocess'):
                post_out = bxr_out
                for post_conv, post_norm, post_relu, post_drop in self._post_blocks:
                    post_conv_out = post_conv(post_out)
                    post_norm_out = post_norm(post_conv_out, training=training)
                    post_relu_out = post_relu(post_norm_out)
                    post_out = post_drop(post_relu_out, training=training)

            with tf.name_scope('logits'):
                return self._last_layer(post_out)

    @staticmethod
    def _build_block(filters, kernel_size, strides, dilation_rate, dropout_rate, conv_cls, sub_block_count=1):
        block_layers = [
            [conv_cls(filters=filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      dilation_rate=dilation_rate,
                      padding='same'),
             BatchNormalization1D(),
             tf.keras.layers.ReLU(),
             tf.keras.layers.Dropout(rate=dropout_rate)] for _ in range(sub_block_count)
        ]

        return block_layers

    def _build_block_with_residual(self, filters, kernel_size, dropout_rate):
        block_layers = self._build_block(filters, kernel_size, 1, 1, dropout_rate, self._conv_cls, self._r)
        block_residual_conv = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, padding='same')
        block_residual_norm = BatchNormalization1D()

        return block_layers, (block_residual_conv, block_residual_norm)

    def get_model_configuration(self):
        return {
            'b': self._b,
            'r': self._r,
            'alphabet': self._alphabet,
            'conv_cls': self._conv_cls,
            'name': self._name,
        }

    @abstractmethod
    def get_layer_configuration(self):
        """Should return a dict containing a detailed model specification.
           Return value must be in the format:

            >>> layer_configuration = {
            >>>     'prep_config': {
            >>>         'filters': 256,
            >>>         'kernel_size': 11,
            >>>         'strides': 2,
            >>>         'dropout_rate': 0.2,
            >>>     },
            >>>     'block_config': {
            >>>         'filters': [256, 384, 512, 640, 768],
            >>>         'kernel_size': [11, 13, 17, 21, 25],
            >>>         'dropout_rate': [0.2, 0.2, 0.2, 0.3, 0.3],
            >>>     },
            >>>     'post_config': {
            >>>         'filters': [896, 1024],
            >>>         'kernel_size': [29, 1],
            >>>         'dropout_rate': [0.4, 0.4],
            >>>     },
            >>> }
        """
        raise NotImplemented

    @staticmethod
    def get_number_of_trainable_variables():
        total_parameters = sum(
            [reduce(lambda x, y: x * y.value, variable.get_shape(), 1) for variable in tf.trainable_variables()])
        return total_parameters
