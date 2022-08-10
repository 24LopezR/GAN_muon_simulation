from re import search
from typing import NamedTuple, Optional
import tensorflow as tf
from tensorflow import (Tensor, TensorShape, concat, one_hot, split, squeeze,
                        stop_gradient, shape, reshape, transpose)
from tensorflow.keras.layers import Activation, Layer
from tensorflow.keras.activations import sigmoid
from tensorflow.nn import softmax

TOL = 1e-8

class AdaptativeStepActivation(Layer):

    def __init__(self, name: Optional[str] = None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.k = tf.constant(100, dtype=tf.float32)

    def call(self, _input):
        _input = softmax(_input)
        smooth_max = tf.reduce_max(_input)
        x = _input / (smooth_max + TOL)
        return tf.minimum(tf.maximum(x - 0.4999, 0) * 10000, 1)