import tensorflow as tf


class Jasper:
    """NVIDIA's Jasper model from https://arxiv.org/pdf/1904.03288.pdf"""

    def __init__(self, b, r):
        self._b = b
        self._r = r

    def __call__(self, inputs, training=None):
        pass

    def get_configuration(self):
        return {
            'b': self._b,
            'r': self._r,
        }
