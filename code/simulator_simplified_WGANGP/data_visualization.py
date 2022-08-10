import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import LearningRateScheduler
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Concatenate
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.initializers import RandomNormal
from keras.constraints import Constraint
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt

from layers.gumbel_softmax import GumbelSoftmaxActivation


def load(inputfile):
    data = pd.read_csv(inputfile).to_numpy()
    in_data = data[:, 0:2]
    first_active = data[:, 2].astype(int)
    last_active = data[:, 3].astype(int)

    n = (last_active - first_active).astype(int)
    cat = 216 * n + first_active - np.vectorize(lambda x: np.sum(range(x)))(n)

    # scale input data (x, vx)
    weights = 1 / np.sqrt(in_data[:, 0] ** 2 + in_data[:, 1] ** 2)
    scaler = StandardScaler()
    scaler.fit(in_data, sample_weight=weights)
    in_data_scaled = scaler.transform(in_data)

    enc = OneHotEncoder(sparse=False)
    enc.fit(cat)
    cat = enc.transform(cat)
    # print(activations.shape)
    # print(activations[0])
    return np.hstack([in_data_scaled, cat]), enc


if __name__ == '__main__':
    [in_data, cat], enc = load(inputfile="/home/ruben/GAN_muon_simulation/data/sim_start_final.csv")

    plt.rcParams["figure.figsize"] = (14, 7)
    plt.rcParams["figure.titlesize"], plt.rcParams["axes.titlesize"] = (20, 20)
    plt.rcParams["axes.labelsize"] = 18
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.hist(in_data[:, 0], bins=60, density=False, histtype='step', color='black')
    ax2.hist(in_data[:, 1], bins=60, density=False, histtype='step', color='black')
    ax1.set_xlabel('$x_0$')
    ax1.set_yscale('log')
    ax2.set_xlabel('$v_{x_0}$')
    ax2.set_yscale('log')
    plt.show()

    plt.hist(cat, bins=1070)
