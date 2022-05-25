############################################################################################
############################################################################################
### Generative Adversarial Network to simulate muon scattering                           ###
###   The input will be random noise and the measurements in the first detector          ###
#######                                                                                  ###
### Use: python cond_gan.py -i [input.root] -e [epochs] -l [latent dim] -k 5             ###
############################################################################################
############################################################################################
import math
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten, Reshape
from keras.layers import LeakyReLU
from keras.layers import Concatenate
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Softmax
from keras.losses import mean_squared_error, kl_divergence
from keras.initializers import RandomUniform, RandomNormal, Constant
from keras.constraints import Constraint
from models.layers.gumbel_softmax import GumbelSoftmaxActivation

# clip model weights to a given hypercube
class ClipConstraint(Constraint):
    # set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value

    # clip model weights to hypercube
    def __call__(self, weights):
        return K.clip(weights, -self.clip_value, self.clip_value)

    # get the config
    def get_config(self):
        return {'clip_value': self.clip_value}


# Model class
class WGAN:

    def __init__(self, input_dim, latent_dim, n_categorical_variables, learning_rate=0.001,
                 lr_drop=1, epochs_drop=25, generator=None, tau=0.2, clip=0.01):
        self.learning_rate = learning_rate
        self.lr_drop = lr_drop
        self.epochs_drop = epochs_drop
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.generator = generator
        self.n_categorical_variables = n_categorical_variables
        self.tau = tau
        self.clip = clip

    # implementation of wasserstein loss
    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    # learning rate controller
    def step_decay(self, epoch):
        initial_lrate = self.learning_rate
        drop = self.lr_drop
        epochs_drop = self.epochs_drop
        lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        return lrate

    # define the standalone critic model
    def define_critic(self):
        # weight clipping
        const = ClipConstraint(self.clip)
        # define model
        input_layer = Input(shape=self.input_dim)
        # data input
        in_data = Input(shape=2)

        critic_n = Concatenate()([input_layer[:, 0:5], in_data])
        critic_n = Dense(100, kernel_constraint=const)(critic_n)
        critic_n = LeakyReLU(alpha=0.2)(critic_n)
        critic_n = Dense(50, kernel_constraint=const)(critic_n)
        critic_n = LeakyReLU(alpha=0.2)(critic_n)
        critic_n = Dense(25, kernel_constraint=const)(critic_n)

        critic_w = Concatenate()([input_layer[:, 5:221], in_data])
        critic_w = Dense(400, kernel_constraint=const)(critic_w)
        critic_w = LeakyReLU(alpha=0.2)(critic_w)
        critic_w = Dense(200, kernel_constraint=const)(critic_w)
        critic_w = LeakyReLU(alpha=0.2)(critic_w)
        critic_w = Dense(100, kernel_constraint=const)(critic_w)
        critic_w = LeakyReLU(alpha=0.2)(critic_w)
        critic_w = Dense(50, kernel_constraint=const)(critic_w)
        critic_w = LeakyReLU(alpha=0.2)(critic_w)

        critic = Concatenate()([critic_n, critic_w])
        critic = Dense(50, kernel_constraint=const)(critic)
        critic = LeakyReLU(alpha=0.2)(critic)

        out_layer = Dense(1)(critic)
        # define model
        model = Model([input_layer, in_data], out_layer)
        # compile model
        opt = Adam(learning_rate=self.learning_rate)
        model.compile(loss=self.wasserstein_loss, optimizer=opt)
        return model

    # define the standalone generator model
    def define_generator(self):
        if self.generator is None:
            # noise input
            in_lat = Input(shape=self.latent_dim)
            # data input
            in_data = Input(shape=2)
            gen = Concatenate()([in_lat, in_data])
            gen = Dense(50)(gen)
            gen = LeakyReLU(alpha=0.2)(gen)
            gen = Dense(100)(gen)
            gen = LeakyReLU(alpha=0.2)(gen)
            gen = Dense(200)(gen)
            gen = LeakyReLU(alpha=0.2)(gen)
            gen = Dense(400)(gen)
            gen = LeakyReLU(alpha=0.2)(gen)
            gen = Dense(400)(gen)
            gen = LeakyReLU(alpha=0.2)(gen)

            n_active = gen[:, 0:50]
            n_active = Dense(5)(n_active)
            n_active = LeakyReLU(alpha=0.2)(n_active)
            n_active = GumbelSoftmaxActivation(n_categories=1, tau=self.tau[0], name='GumbelSoftmax_N')(n_active)

            first_active = gen[:, 50:400]
            first_active = Dense(350)(first_active)
            first_active = LeakyReLU(alpha=0.2)(first_active)
            first_active = Dense(216)(first_active)
            first_active = LeakyReLU(alpha=0.2)(first_active)
            first_active = GumbelSoftmaxActivation(n_categories=1, tau=self.tau[1], name='GumbelSoftmax_W')(
                first_active)
            # output
            out_layer = Concatenate()([n_active, first_active])
            # define model
            model = Model([in_lat, in_data], out_layer)
        else:
            model = self.generator
        return model

    # define the combined generator and discriminator model, for updating the generator
    def define_gan(self):
        # create models
        critic = self.define_critic()
        generator = self.define_generator()
        # make weights in the critic not trainable
        critic.trainable = False
        # get noise from generator model
        [gen_noise, in_data] = generator.input
        # get image output from the generator model
        gen_output = generator.output
        # connect image output and label input from generator as inputs to discriminator
        gan_output = critic([gen_output, in_data])
        # define gan model as taking noise and label and outputting a classification
        gan_model = Model([gen_noise, in_data], gan_output)
        # compile model
        opt = Adam(learning_rate=self.learning_rate)
        gan_model.compile(loss=self.wasserstein_loss, optimizer=opt)
        self.critic = critic
        self.generator = generator
        self.gan = gan_model

    def penalty(x):
        x = K.reshape(x, (216, 2))
        x = K.argmax(x)
        TOL = 0.05
        A = 0.1
        s = K.mean(K.sum(x))
        return A * K.exp(1 / (s + TOL))
