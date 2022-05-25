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
from tensorflow.keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LeakyReLU
from keras.layers import Concatenate
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.losses import mean_squared_error
from keras.initializers import RandomUniform, RandomNormal, Zeros
from keras.constraints import Constraint

def heaviside(x):
    k = 1e6
    return 0.5*(1 + K.tanh(k*x))

def penalty(x):
    C = 1e-4
    pen = C * (1 - K.sum(x))**2
    return pen

# implementation of wasserstein loss
def wasserstein_loss(y_true, y_pred):
	return K.mean(y_true * y_pred)

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

# learning rate controller
def step_decay(epoch):
	initial_lrate = 5e-4
	drop = 0.75
	epochs_drop = 25
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

# define the standalone discriminator model
def define_discriminator(n_wires=216):
    # weight constraint
    const = ClipConstraint(0.01)
    # activations input
    in_activations = Input(shape=n_wires)
    fe = Dense(n_wires*2, kernel_constraint=const)(in_activations)
    fe = LeakyReLU(alpha=0.2)(fe)
    #fe = Dropout(0.4)(fe)
    fe = BatchNormalization()(fe)
    fe = Dense(n_wires*2, kernel_constraint=const)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    #fe = Dropout(0.4)(fe)
    fe = BatchNormalization()(fe)
    fe = Dense(n_wires, kernel_constraint=const)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    #fe = Dropout(0.4)(fe)
    #fe = BatchNormalization()(fe)
    #fe = Dense(n_wires, kernel_constraint=const)(fe)
    #fe = LeakyReLU(alpha=0.2)(fe)
    #fe = Dropout(0.4)(fe)
    fe = BatchNormalization()(fe)
    # output
    out_layer = Dense(1, activation='linear')(fe)
    # define model
    model = Model(in_activations, out_layer)
    # compile model
    opt = RMSprop(lr=0.00005)
    #opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    model.compile(loss=wasserstein_loss, optimizer=opt)
    return model

# define the standalone generator model
def define_generator(latent_dim, n_wires=216):
    act_fun_out = heaviside
    # noise input
    in_latent = Input(shape=latent_dim)
    gen = Dense(n_wires*4)(in_latent)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Dense(n_wires*2)(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Dense(n_wires*2)(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Dense(n_wires)(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Dense(n_wires)(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
	# output
    out_layer = Dense(n_wires, activation=act_fun_out, bias_initializer=RandomNormal(mean=0, stddev=1))(gen)
	# define model
    model = Model(in_latent, out_layer)
    return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
    # make weights in the critic not trainable
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    # get noise and data inputs from generator model
    gen_noise = g_model.input
    # get image output from the generator model
    gen_output = g_model.output
    # connect image output and label input from generator as inputs to discriminator
    gan_output = d_model(gen_output)
    # define gan model as taking noise and label and outputting a classification
    model = Model(gen_noise, gan_output)
    #model.add_loss(penalty(gen_output))
    # compile model
    opt = RMSprop(lr=0.00005)
    #opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    model.compile(loss=wasserstein_loss, optimizer=opt)
    return model
