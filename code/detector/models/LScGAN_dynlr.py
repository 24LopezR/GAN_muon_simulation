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
from tensorflow.keras.optimizers import Adam
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LeakyReLU
from keras.layers import Concatenate
from keras.layers import Activation
from keras.layers import Discretization
from keras.layers import Dropout
from keras.losses import mean_squared_error, hinge

# Custom loss function
def custom_loss(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    penalty = 0.
    return mse + penalty

# learning rate controller
def step_decay(epoch):
	initial_lrate = 0.001
	drop = 0.5
	epochs_drop = 25
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

# define the standalone discriminator model
def define_discriminator(in_shape=4):
	# first detector input
	in_first_det_data = Input(shape=in_shape)
	# second detector input
	in_second_det_data = Input(shape=in_shape)
	# concat label as a channel
	merge = Concatenate()([in_second_det_data, in_first_det_data])

	fe = Dense(64)(merge)
	fe = LeakyReLU(alpha=0.2)(fe)
	fe = Dropout(0.4)(fe)
	fe = Dense(64)(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	fe = Dropout(0.4)(fe)
	fe = Dense(32)(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	# output
	out_layer = Dense(1, activation='linear')(fe)
	# define model
	model = Model([in_second_det_data, in_first_det_data], out_layer)
	# compile model
	opt = Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
	model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
	return model

# define the standalone generator model
def define_generator(latent_dim):
	# data input
    in_first_detector = Input(shape=4)
	# noise input
    in_lat = Input(shape=latent_dim)
    merge = Concatenate()([in_lat, in_first_detector])
    gen = Dense(512)(merge)
    gen = Activation('relu')(gen)

    gen = Dense(512)(gen)
    gen = Activation('relu')(gen)
    gen = Dense(256)(gen)
    gen = Activation('relu')(gen)

    gen = Dense(256)(gen)
    gen = Activation('relu')(gen)
    gen = Dense(128)(gen)
    gen = Activation('relu')(gen)
    gen = Dense(128)(gen)
    gen = Activation('relu')(gen)
    gen = Dense(64)(gen)
    gen = Activation('relu')(gen)
	# output
    out_layer = Dense(4, activation='linear')(gen)
	# define model
    model = Model([in_lat, in_first_detector], out_layer)
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
	opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
	model.compile(loss='mse', optimizer=opt, run_eagerly=True)
	return model
