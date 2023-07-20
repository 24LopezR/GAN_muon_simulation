############################################################################################
############################################################################################
### Generative Adversarial Network to simulate muon scattering                           ###
###   The input will be random noise and the measurements in the first detector          ###
#######                                                                                  ###
### Use: python cond_gan.py -i [input.root] -e [epochs] -l [latent dim] -k 5             ###
############################################################################################
############################################################################################
import numpy as np
import tensorflow as tf
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
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
from matplotlib import pyplot
from optparse import OptionParser
import ROOT as r
from array import array
from sklearn.preprocessing import StandardScaler
from keras import backend
from keras.layers import BatchNormalization
from keras.initializers import RandomNormal
from keras.constraints import Constraint

# Wasserstein GAN implementation #########################################
WGAN = True
# implementation of wasserstein loss
def wasserstein_loss(y_true, y_pred):
	return backend.mean(y_true * y_pred)

# clip model weights to a given hypercube
class ClipConstraint(Constraint):
	# set clip value when initialized
	def __init__(self, clip_value):
		self.clip_value = clip_value
 
	# clip model weights to hypercube
	def __call__(self, weights):
		return backend.clip(weights, -self.clip_value, self.clip_value)
 
	# get the config
	def get_config(self):
		return {'clip_value': self.clip_value}
##########################################################################

def step_decay(epoch):
   initial_lrate = 0.1
   drop = 0.1
   epochs_drop = 25
   lrate = initial_lrate * math.pow(drop,  
           math.floor((1+epoch)/epochs_drop))
   return initial_lrate

# define the standalone discriminator model
def define_discriminator(in_shape=4):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# weight constraint
	const = ClipConstraint(0.01)
	
	# first detector input
	in_first_det_data = Input(shape=in_shape)
	# second detector input
	in_second_det_data = Input(shape=in_shape)
	# concat label as a channel
	merge = Concatenate()([in_second_det_data, in_first_det_data])
	
	fe = Dense(128, kernel_initializer=init, kernel_constraint=const)(merge)
	fe = LeakyReLU(alpha=0.2)(fe)
	
	fe = Dense(64, kernel_initializer=init, kernel_constraint=const)(fe)
	fe = LeakyReLU(alpha=0.2)(fe)

	fe = Dense(64, kernel_initializer=init, kernel_constraint=const)(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	#fe = BatchNormalization()(fe)

	fe = Dense(64, kernel_initializer=init, kernel_constraint=const)(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	#fe = BatchNormalization()(fe)
	
	#fe = Flatten()(fe)
	# output
	out_layer = Dense(1, activation='linear')(fe)
	# define model
	model = Model([in_second_det_data, in_first_det_data], out_layer)
	# compile model
	opt = RMSprop(lr=0.001)
	model.compile(loss=wasserstein_loss, optimizer=opt, metrics=['accuracy'])
	return model
 
# define the standalone generator model
def define_generator(latent_dim):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	
	# data input
	in_first_detector = Input(shape=4)
	# noise input
	in_lat = Input(shape=latent_dim)

	n_nodes = 4 * 127
	gen = Dense(n_nodes, kernel_initializer=init)(in_lat)

	merge = Concatenate()([gen, in_first_detector])
	gen = Dense(512)(merge)
	gen = Activation('relu')(gen)
	#gen = BatchNormalization()(gen)
	gen = Dense(256)(gen)
	gen = Activation('relu')(gen)
	#gen = BatchNormalization()(gen)
	gen = Dense(256)(gen)
	gen = Activation('relu')(gen)
	gen = Dense(128)(gen)
	gen = Activation('relu')(gen)
	gen = Dense(64)(gen)
	gen = Activation('relu')(gen)
	gen = Dense(16)(gen)
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
	opt = RMSprop(lr=0.001)
	model.compile(loss=wasserstein_loss, optimizer=opt)
	return model
