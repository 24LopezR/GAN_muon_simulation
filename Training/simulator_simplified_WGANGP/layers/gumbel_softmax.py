"""Gumbel-Softmax layer implementation.
Reference: https://arxiv.org/pdf/1611.04051.pdf"""
from re import search
from typing import NamedTuple, Optional

# pylint: disable=E0401
from tensorflow import (Tensor, TensorShape, concat, one_hot, split, squeeze,
                        stop_gradient, shape, reshape, transpose)
from tensorflow.keras.layers import Activation, Layer
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.math import log, reduce_sum
from tensorflow.nn import softmax
from tensorflow.random import categorical, uniform
import tensorflow_probability as tfp

TOL = 1e-20

def gumbel_noise(shape: TensorShape):
    """Create a single sample from the standard (loc = 0, scale = 1) Gumbel distribution."""
    uniform_sample = uniform(shape, minval=0, maxval=1)
    return -log(-log(uniform_sample + TOL) + TOL)


class GumbelSoftmaxLayer(Layer):
    "A Gumbel-Softmax layer implementation that should be stacked on top of a categorical feature logits."

    def __init__(self, tau: float = 0.2, name: Optional[str] = None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.tau = tau

    def call(self, _input):
        """Computes Gumbel-Softmax for the logits output of a particular categorical feature."""
        noised_input = _input + gumbel_noise(shape(_input))
        soft_sample = softmax(logits=noised_input, axis=1)
        return soft_sample

    def get_config(self):
        config = super().get_config().copy()
        config.update({'tau': self.tau})
        return config


class GumbelSoftmaxActivation(Layer):
    """An interface layer connecting different parts of an incoming tensor to adequate activation functions.
    The tensor parts are qualified according to the passed processor object.
    Processed categorical features are sent to specific Gumbel-Softmax layers.
    Processed features of different kind are sent to a TanH activation.
    Finally all output parts are concatenated and returned in the same order.
    The parts of an incoming tensor are qualified by leveraging a namedtuple pointing to each of the used data \
        processor's pipelines in/out feature maps. For simplicity this object can be taken directly from the data \
        processor col_transform_info."""

    def __init__(self, n_categorical_variables: int = 2, tau: float = 0.2, name: Optional[str] = None, **kwargs):
        """Arguments:
            col_map (NamedTuple): Defines each of the processor pipelines input/output features.
            name (Optional[str]): Name of the layer"""
        super().__init__(name=name, **kwargs)
        self.tau          = tau
        self.n_categorical_variables = n_categorical_variables

    def call(self, _input):  # pylint: disable=W0221
        cat_cols = split(_input, self.n_categorical_variables, axis=1)
        cat_cols_gs = [GumbelSoftmaxLayer(tau=self.tau)(col) for col in cat_cols]
        return concat(cat_cols_gs, axis=1)

    def get_config(self):
        config = super().get_config().copy()
        return config

if __name__=='__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf
    import tensorflow_probability as tfp

    n = 20
    a = np.random.uniform(size=n)
    a[15] += 2
    a = tf.constant(a, shape=(1,n))
    x = np.linspace(1,n,n)
    plt.scatter(x,a)
    plt.show()

    ags = []
    arohc = []
    for i in range(1):
        ags_temp, hard_sample = GumbelSoftmaxLayer(tau=0.1)(a)
        ags.append(tf.argmax(ags_temp, axis=1))
        arohc_temp = tfp.distributions.RelaxedOneHotCategorical(0.1, probs=a).sample()
        arohc.append(tf.argmax(arohc_temp, axis=1))

    plt.scatter(x, np.asarray(ags_temp), color='red')
    plt.scatter(x, np.asarray(hard_sample), color='pink', marker='*')
    plt.scatter(x, np.asarray(arohc_temp), color='black')
    plt.show()