##################################################################################################
#### Exectubale script to train and evaluate a GAN                                            ####
##################################################################################################
import numpy as np
import tensorflow as tf
from keras import backend
from keras.models import load_model
from optparse import OptionParser
from sklearn.preprocessing import StandardScaler
import importlib
import os

from utils.dataloader import load, scale
from utils.evaluating import evaluate
from utils.generating import generate_and_save, generate_evaluation_samples
from utils.training import train
from utils.plotting import print_plots

if __name__ == "__main__":
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

	# _________________________________________________________________________________________
	# Read user options
	parser = OptionParser(usage="%prog --help")
	parser.add_option("-g", "--gan",       dest="gan_model",   type="string",   default='LScGAN',         help="GAN model architecture")
	parser.add_option("-i", "--input",     dest="input",       type="string",   default='input.root',     help="Input root filename")
	parser.add_option("-o", "--output",    dest="output",      type="string",   default='output',         help="Output filename")
	parser.add_option("-e", "--epochs",    dest="epochs",      type="int",      default=100,              help="N epochs")
	parser.add_option("-l", "--latent",    dest="latent",      type="int",      default=64,               help="Dimension of latent space")
	parser.add_option("-b", "--batch",     dest="batch",       type="int",      default=256,              help="N batch")
	parser.add_option("-k", "--khyp",      dest="k_hyp",       type="int",      default=5,                help="k hyperparameter")
	parser.add_option("-m", "--init",      dest="init",        type="string",   default=None,        help="Initial generator model")
	parser.add_option("-t", "--no-train",  dest="train",       action="store_false")
	(options, args) = parser.parse_args()

	# GAN model
	gan_m = options.gan_model
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

	# initial gen model
	g_model = None
	if options.init is not None:
		g_model_name = options.init
		g_model = load_model(g_model_name)
	print('_'*90)
	print('OPTIONS')
	print('_'*90)
	print(' GAN model: {:<50}'.format(options.gan_model))
	print(' Training data file: {:<50}'.format(options.input))
	print(' Output model name: {:<50}'.format(options.output))
	print(' Training mode: {:<50}'.format('False' if options.train is False else 'True'))
	if options.train is not False:
		print('     Training epochs: {:<50}'.format(str(options.epochs)))
		print('     Dimension of latent space: {:<50}'.format(str(options.latent)))
		print('     Batch size: {:<50}'.format(str(options.batch)))
		print('     k hyperparameter: {:<50}'.format(str(options.k_hyp)))
	if options.init is not None:
		print(' Initial generator model: {:<50}'.format(options.init))
	print('_'*90)
	# _________________________________________________________________________________________

	# GPU memory usage configuration
	config = tf.compat.v1.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.compat.v1.Session(config=config)
	backend.set_session(sess)
	# _________________________________________________________________________________________

	# load and scale data
	scaler = StandardScaler()
	data = load(inputfile, measurements=False)
	dataset, _, scaler = scale(data, scaler)
	# _________________________________________________________________________________________

	# Create and train model
	if options.train is not False:
		GAN = importlib.import_module('models.'+gan_m)
		# create the discriminator
		d_model = GAN.define_discriminator()
		d_model.summary()
		# create the generator
		if options.init is None:
			g_model = GAN.define_generator(latent_dim)
		g_model.summary()
		# create the gan
		gan_model = GAN.define_gan(g_model, d_model)
		gan_model.summary()
		g_model = train(g_model, d_model, gan_model, dataset, latent_dim, GAN.step_decay, n_discrim_updates, epochs, nbatch)
	# _________________________________________________________________________________________
	
	# Evaluate the model
	if g_model is not None:
		fake = generate_evaluation_samples(g_model, dataset, latent_dim, scaler)
		evaluate(data[:,4:], fake)
	else:
		fake = None
		print("."*90)
		print('Warning: There is no generator model to evaluate')
		print("."*90)
	print_plots(data, fake, output)
	# _________________________________________________________________________________________
	
	#Generate a few
	#generate_and_save(g_model, dataset, latent_dim, 300000, scaler, output)
	# _________________________________________________________________________________________
