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
from models.layers.gumbel_softmax import GumbelSoftmaxLayer

# Model parameters
CLIP = 0.01
LEARNING_RATE = 0.00005
DROP = 0.9
EPOCHS_DROP = 25

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

# implementation of wasserstein loss
def wasserstein_loss(y_true, y_pred):
	return K.mean(y_true * y_pred)
    
# learning rate controller
def step_decay(epoch):
	initial_lrate = LEARNING_RATE
	drop = DROP
	epochs_drop = EPOCHS_DROP
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

# define the standalone critic model
def define_critic(input_dim=216):
    # weight constraint
    const = ClipConstraint(CLIP)
    # define model
    input_layer = Input((input_dim,))
    critic = Dense(432, kernel_constraint=const)(input_layer)
    critic = LeakyReLU(alpha=0.2)(critic)
    critic = Dense(216, kernel_constraint=const)(critic)
    critic = LeakyReLU(alpha=0.2)(critic)
    critic = Dense(216, kernel_constraint=const)(critic)
    critic = LeakyReLU(alpha=0.2)(critic)
    critic = Dense(54, kernel_constraint=const)(critic)
    critic = LeakyReLU(alpha=0.2)(critic)
    out_layer = Dense(1)(critic)
    # define model
    model = Model(input_layer, out_layer)
    # compile model
    opt = RMSprop(lr=LEARNING_RATE)
    model.compile(loss=wasserstein_loss, optimizer=opt)
    return model

# define the standalone generator model
def define_generator(latent_dim, input_dim=216):
    # noise input
    in_lat = Input(shape=latent_dim)
    gen = Dense(1080)(in_lat)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Dense(864)(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Dense(864)(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Dense(432)(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
	# output
    gen = Dense(216)(gen)
    out_layer_soft = GumbelSoftmaxLayer(tau=0.2, name='GSoftmax')(gen)
	# define model
    model = Model(in_lat, out_layer_soft)
    return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
    # make weights in the critic not trainable
    d_model.trainable = False
    # get noise from generator model
    gen_noise = g_model.input
    # get image output from the generator model
    gen_output = g_model.output
    # connect image output and label input from generator as inputs to discriminator
    gan_output = d_model(gen_output)
    # define gan model as taking noise and label and outputting a classification
    model = Model(gen_noise, gan_output)
    # compile model
    opt = RMSprop(lr=LEARNING_RATE)
    model.compile(loss=wasserstein_loss, optimizer=opt)
    return model
