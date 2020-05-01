import math
from functools import reduce

import tensorflow as tf

from layers.batch_normalization import BatchNormalization1D


class Jasper:
    """NVIDIA's Jasper model from https://arxiv.org/pdf/1904.03288.pdf"""

    def __init__(self, b, r, alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ\''):
        self._b = b
        self._r = r
        self._alphabet = alphabet + '*'  # Add 'blank' symbol

        # Preprocess block layers
        self._prep_conv, self._prep_norm, self._prep_relu, self._prep_drop = self._build_block(filters=256,
                                                                                               kernel_size=11,
                                                                                               strides=2,
                                                                                               dilation_rate=1,
                                                                                               dropout_rate=0.2)[0]

        # Block parameters
        self._block_filters = [256, 384, 512, 640, 768]
        self._block_kernels = [11, 13, 17, 21, 25]
        self._block_dropouts = [0.2, 0.2, 0.2, 0.3, 0.3]
        self._blocks = [self._build_block_with_residual(filters=f, kernel_size=k, dropout_rate=d)
                        for f, k, d in zip(self._block_filters, self._block_kernels, self._block_dropouts)]

        # Postprocess block layers
        post_a = self._build_block(filters=896, kernel_size=29, strides=1, dilation_rate=2, dropout_rate=0.4)
        post_b = self._build_block(filters=1024, kernel_size=1, strides=1, dilation_rate=1, dropout_rate=0.4)
        self._post_blocks = [post_a[0], post_b[0]]

        # Last layer (alphabet + 'blank')
        self._last_layer = tf.keras.layers.Conv1D(filters=len(self._alphabet), kernel_size=1)

    def __call__(self, inputs, training=None):
        with tf.name_scope('Jasper'):
            with tf.name_scope('Preprocess'):
                prep_conv = self._prep_conv(inputs)
                prep_norm = self._prep_norm(prep_conv, training=training)
                prep_relu = self._prep_relu(prep_norm)
                prep_out = self._prep_drop(prep_relu, training=training)

            with tf.name_scope(f'{self._b}x{self._r}'):
                bxr_out = prep_out
                for layers, res_layers in self._blocks:
                    res_conv, res_norm = res_layers
                    for conv, norm, relu, drop in layers:
                        conv_out = conv(bxr_out)
                        norm_out = norm(conv_out, training=training)
                        relu_out = relu(norm_out)
                        bxr_out = drop(relu_out)

                    res_conv_out = res_conv(bxr_out)
                    bxr_out = res_norm(res_conv_out, training=training)

            with tf.name_scope('Postprocessing'):
                post_out = bxr_out
                for post_conv, post_norm, post_relu, post_drop in self._post_blocks:
                    post_conv_out = post_conv(post_out)
                    post_norm_out = post_norm(post_conv_out, training=training)
                    post_relu_out = post_relu(post_norm_out)
                    post_out = post_drop(post_relu_out, training=training)

            with tf.name_scope('logits'):
                return self._last_layer(post_out)

    @staticmethod
    def _build_block(filters, kernel_size, strides, dilation_rate, dropout_rate, sub_block_count=1):
        block_layers = [
            [tf.keras.layers.Conv1D(filters=filters,
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
        block_layers = self._build_block(filters, kernel_size, 1, 1, dropout_rate, self._r)
        block_residual_conv = tf.keras.layers.Conv1D(filters=filters,
                                                     kernel_size=kernel_size,
                                                     padding='same')
        block_residual_norm = BatchNormalization1D()

        return block_layers, (block_residual_conv, block_residual_norm)

    def get_configuration(self):
        return {
            'b': self._b,
            'r': self._r,
            'alphabet': self._alphabet,
        }

    @staticmethod
    def get_number_of_trainable_variables():
        total_parameters = sum(
            [reduce(lambda x, y: x * y.value, variable.get_shape(), 1) for variable in tf.trainable_variables()])
        return total_parameters
