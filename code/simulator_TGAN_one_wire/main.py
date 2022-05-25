##################################################################################################
#### Exectubale script to train and evaluate a GAN                                            ####
#### that will simulate the Detector part                                                     ####
##################################################################################################
import numpy as np
import tensorflow as tf
from keras import backend
from keras.models import load_model
from optparse import OptionParser
import importlib
import os
import sys
sys.path.append('/home/ruben/Documents/TFM/GAN_muon_simulation/code/simulator_TGAN')
from utils.dataloader import load, scale
from utils.evaluating import evaluate
from utils.generating_WGAN import generate_evaluation_samples
from utils.training_WGAN import train
from utils.plotting import print_plots
import utils.cfg as conf
from models.layers.gumbel_softmax import GumbelSoftmaxLayer

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # _________________________________________________________________________________________
    # Read user options
    parser = OptionParser(usage="%prog --help")
    parser.add_option("-g", "--gan",       dest="gan_model",   type="string",   default='WGAN',                                                              help="GAN model architecture")
    parser.add_option("-i", "--input",     dest="input",       type="string",   default='/home/ruben/Documents/TFM/GAN_muon_simulation/data/file_2.csv',     help="Input root filename")
    parser.add_option("-o", "--output",    dest="output",      type="string",   default='output',                                                            help="Output filename")
    parser.add_option("-e", "--epochs",    dest="epochs",      type="int",      default=200,                                                                 help="N epochs")
    parser.add_option("-l", "--latent",    dest="latent",      type="int",      default=432,                                                                 help="input_dim of latent space")
    parser.add_option("-b", "--batch",     dest="batch",       type="int",      default=256,                                                                 help="N batch")
    parser.add_option("-k", "--khyp",      dest="k_hyp",       type="int",      default=5,                                                                   help="k hyperparameter")
    parser.add_option("-u", "--unrolling", dest="unrolling",   type="int",      default=1,                                                                   help="unrolling hyperparameter")
    parser.add_option("-lr","--learning-rate", dest="lr",      type="float",    default=0.00005,                                                             help="Learning rate")
    parser.add_option("-m", "--init",      dest="init",        type="string",   default=None,                                                                help="Initial generator model")
    parser.add_option("-t", "--no-train",  dest="train",       action="store_false")
    (options, args) = parser.parse_args()

	# GAN model
    gan_m = options.gan_model
    if gan_m == 'LScGAN_dynlr': conf.CONDITIONAL = True
	# Name of input file
    inputfile = options.input
	# Name of output file
    output = options.output
	# number of epochs
    epochs = options.epochs
	# size of the latent space
    latent_dim = options.latent
	# batch size
    nbatch = options.batch
	# number or discriminator updates
    n_discrim_updates = options.k_hyp
    # number or generator updates
    n_gen_updates = options.unrolling
    # learning rate
    learning_rate = options.lr

	# initial gen model
    g_model = None
    if options.init is not None:
        g_model_name = options.init
        g_model = load_model(g_model_name, custom_objects={"GumbelSoftmaxLayer":GumbelSoftmaxLayer})
    print('_'*90)
    print('OPTIONS')
    print('_'*90)
    print(' GAN model: {:<50}'.format(options.gan_model))
    print(' Training data file: {:<50}'.format(options.input))
    print(' Output model name: {:<50}'.format(options.output))
    print(' Training mode: {:<50}'.format(
	    'False' if options.train is False else 'True'))
    if options.train is not False:
        print('     Training epochs: {:<50}'.format(str(options.epochs)))
        print('     input_dim of latent space: {:<50}'.format(str(options.latent)))
        print('     Batch size: {:<50}'.format(str(options.batch)))
        print('     k hyperparameter: {:<50}'.format(str(options.k_hyp)))
        print('     Unrolling hyperparameter: {:<50}'.format(str(options.unrolling)))
        print('     Learning rate: {:<50}'.format(str(options.lr)))
    if options.init is not None:
        print(' Initial generator model: {:<50}'.format(options.init))
        g_model.summary()
    print('_'*90)
	# _________________________________________________________________________________________

	# GPU memory usage configuration
    cfg = tf.compat.v1.ConfigProto()
    cfg.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=cfg)
    backend.set_session(sess)
	# _________________________________________________________________________________________

	# load and scale data
    dataset = load(inputfile)
    dataset = scale(dataset)
    input_dim = 216
	# _________________________________________________________________________________________

	# Create and train model
    if options.train is not False:
        GAN = importlib.import_module('models.'+gan_m)
		# create the discriminator
        d_model = GAN.define_critic(input_dim)
        d_model.summary()
		# create the generator
        if options.init is None:
            g_model = GAN.define_generator(latent_dim, input_dim=input_dim)
        g_model.summary()
		# create the gan
        gan_model = GAN.define_gan(g_model, d_model)
        gan_model.summary()
        g_model = train(g_model, d_model, gan_model, dataset, latent_dim, GAN.step_decay, n_discrim_updates, n_gen_updates, epochs, nbatch)
	# _________________________________________________________________________________________
	
	# Evaluate the model
    if g_model is not None:
        activations, activations_gen = generate_evaluation_samples(g_model, dataset, latent_dim)
        print_plots(activations, activations_gen)
        evaluate(activations, activations_gen)
    else:
        fake = None
        print("."*90)
        print('Warning: There is no generator model to evaluate')
        print("."*90)
	# _________________________________________________________________________________________
	
	# Generate a few
	# generate_and_save(g_model, dataset, latent_dim, 300000, encoder, output)
	# _________________________________________________________________________________________