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
from keras.layers import Dropout
from keras.losses import mean_squared_error
from keras.initializers import RandomUniform

def mod_sigmoid(x):
    return (K.sigmoid(x))

def mod_mse(y_true, y_pred):
    MSE = mean_squared_error(y_true, y_pred)
    Dhits = (K.sum(K.round(y_pred), axis=1)/K.sum(y_true, axis=1))
    print(y_true.shape)
    print(y_pred.shape)
    y_true = K.print_tensor(y_true, message='ytrue = ')
    y_pred = K.print_tensor(y_pred, message='ypred = ')
    MSE = K.print_tensor(MSE, message='MSE = ')
    Dhits = K.print_tensor(Dhits, message='D(hits) = ')
    return MSE + Dhits

# learning rate controller
def step_decay(epoch):
	initial_lrate = 0.0001
	drop = 0.5
	epochs_drop = 25
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

# define the standalone discriminator model
def define_discriminator(n_wires=216):
    # activations input
    in_activations = Input(shape=n_wires)
    fe = Dense(216*4)(in_activations)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.4)(fe)
    fe = Dense(216*4)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.4)(fe)
    fe = Dense(108*4)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.4)(fe)
    fe = Dense(54*4)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.4)(fe)
    fe = Dense(27*4)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    # output
    out_layer = Dense(1, activation='linear')(fe)
    # define model
    model = Model(in_activations, out_layer)
    # compile model
    opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
    return model

# define the standalone generator model
def define_generator(latent_dim, n_wires=216):
    act_fun = 'sigmoid'
    # noise input
    in_lat = Input(shape=latent_dim)
    gen = Dense(n_wires*16)(in_lat)
    gen = Activation(act_fun)(gen)
    gen = Dense(n_wires*16)(gen)
    gen = Activation(act_fun)(gen)
    gen = Dense(n_wires*16)(gen)
    gen = Activation(act_fun)(gen)
    gen = Dense(n_wires*8)(gen)
    gen = Activation(act_fun)(gen)
    gen = Dense(n_wires*8)(gen)
    gen = Activation(act_fun)(gen)
    gen = Dense(n_wires*4)(gen)
    gen = Activation(act_fun)(gen)
	# output
    out_layer = Dense(n_wires, activation=act_fun, kernel_initializer=RandomUniform(),
                      bias_initializer="zeros")(gen)
	# define model
    model = Model(in_lat, out_layer)
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
    opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    model.compile(loss='mse', optimizer=opt)
    return model
