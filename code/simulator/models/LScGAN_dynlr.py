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

def heaviside(x):
    k = 1e6
    return 0.5*(1 + K.tanh(k*x))

def penalty(x):
    C = 1e-4
    pen = C * (1 - K.sum(x))**2
    return pen

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
	initial_lrate = 1e-4
	drop = 0.5
	epochs_drop = 25
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

# define the standalone discriminator model
def define_discriminator(n_wires=216):
    # data input
    in_data = Input(shape=2)
    in_dat = Dense(n_wires)(in_data)
    # activations input
    in_activations = Input(shape=n_wires)
    in_act = Dense(n_wires, activation='sigmoid')(in_activations)
    # concat label as a channel
    merge = Concatenate()([in_act, in_dat])

    fe = Dense(n_wires*2)(merge)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.4)(fe)
    fe = BatchNormalization()(fe)
    fe = Dense(n_wires*2)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.4)(fe)
    fe = BatchNormalization()(fe)
    fe = Dense(n_wires)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.4)(fe)
    fe = BatchNormalization()(fe)
    fe = Dense(n_wires)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Dropout(0.4)(fe)
    fe = BatchNormalization()(fe)
    # output
    out_layer = Dense(1, activation='linear')(fe)
    # define model
    model = Model([in_activations, in_data], out_layer)
    #model.add_loss(penalty(out_layer))
    # compile model
    #opt = RMSprop()
    opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
    return model

# define the standalone generator model
def define_generator(latent_dim, n_wires=216):
    act_fun = 'relu'
    act_fun_out = heaviside
    # noise input
    in_latent = Input(shape=latent_dim)
    in_lat = Dense(n_wires)(in_latent)
	# data input
    in_data = Input(shape=2)
    in_dat = Dense(n_wires)(in_data)
    merge = Concatenate()([in_lat, in_dat])
    gen = Dense(n_wires*4)(merge)
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
    out_layer = Dense(n_wires, activation=act_fun_out, bias_initializer=RandomNormal(mean=-0.3, stddev=0.1))(gen)
	# define model
    model = Model([in_latent, in_data], out_layer)
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
    model.add_loss(penalty(gen_output))
    # compile model
    #opt = RMSprop()
    opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    model.compile(loss='mse', optimizer=opt)
    return model
