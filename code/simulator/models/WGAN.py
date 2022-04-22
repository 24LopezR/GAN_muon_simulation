############################################################################################
############################################################################################
### Generative Adversarial Network to simulate muon scattering                           ###
#######                                                                                  ###
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
from keras.losses import mean_squared_error
from keras.constraints import Constraint

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

def mod_wasserstein(y_true, y_pred):
    return K.mean(y_true*y_pred) + (K.sum(y_true)-K.sum(y_pred))**2

# learning rate controller
def step_decay(epoch):
	initial_lrate = 0.00005
	drop = 0.5
	epochs_drop = 25
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

# define the standalone discriminator model
def define_discriminator(n_wires=216):
    # weight constraint
    const = ClipConstraint(0.01)
    # data input
    in_data = Input(shape=2)
    # activations input
    in_activations = Input(shape=n_wires)
    in_act = Dense(108, activation='sigmoid')(in_activations)
    # concat label as a channel
    merge = Concatenate()([in_act, in_data])

    fe = Dense(110, kernel_constraint=const)(merge)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.4)(fe)
    fe = Dense(55, kernel_constraint=const)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dense(11, kernel_constraint=const)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dense(11, kernel_constraint=const)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    # output
    out_layer = Dense(1, activation='linear')(fe)
    # define model
    model = Model([in_activations, in_data], out_layer)
    # compile model
    opt = RMSprop(lr=0.00005)
    model.compile(loss=mod_wasserstein, optimizer=opt, metrics=['accuracy'])
    return model

# define the standalone generator model
def define_generator(latent_dim, n_wires=216):
    act_fun = 'sigmoid'
    # noise input
    in_lat = Input(shape=latent_dim)
	# data input
    in_data = Input(shape=2)
    merge = Concatenate()([in_lat, in_data])
    gen = Dense(n_wires*4)(merge)
    gen = Activation(act_fun)(gen)
    gen = Dense(n_wires*4)(gen)
    gen = Activation(act_fun)(gen)
    gen = Dense(n_wires*2)(gen)
    gen = Activation(act_fun)(gen)
    gen = Dense(n_wires*2)(gen)
    gen = Activation(act_fun)(gen)
    gen = Dense(n_wires)(gen)
    gen = Activation(act_fun)(gen)
	# output
    out_layer = Dense(n_wires, activation=act_fun, kernel_initializer="zeros",
                      bias_initializer="zeros")(gen)
	# define model
    model = Model([in_lat, in_data], out_layer)
    return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
	# make weights in the critic not trainable
	d_model.trainable = False
	# get noise and data inputs from generator model
	gen_noise, gen_data_input = g_model.input
	# get image output from the generator model
	gen_output = g_model.output
	# connect image output and label input from generator as inputs to discriminator
	gan_output = d_model([gen_output, gen_data_input])
	# define gan model as taking noise and label and outputting a classification
	model = Model([gen_noise, gen_data_input], gan_output)
	# compile model
	opt = RMSprop(lr=0.00005)
	model.compile(loss=mod_wasserstein, optimizer=opt)
	return model