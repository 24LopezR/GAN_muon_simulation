import numpy as np
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
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
from matplotlib import pyplot
from optparse import OptionParser
import ROOT as r
from array import array
from sklearn.preprocessing import StandardScaler
from keras.models import load_model

# import my functions
from cGAN import load_real_samples, generate_latent_points, generate_fake_samples, generate_and_save

##################################################################
##################################################################
########################### Main #################################
##################################################################
##################################################################
if __name__ == "__main__":

	parser = OptionParser(usage="%prog --help")
	parser.add_option("-i", "--input",     dest="input",       type="string",   default='input.root',     help="Input root file")
	parser.add_option("-m", "--model",     dest="model",       type="string",   default='moddel.h5',      help="Model .h5 file")
	parser.add_option("-o", "--output",    dest="output",      type="string",   default='output.root',    help="Output root file")
	parser.add_option("-n", "--nsamples",  dest="samples",     type="int",      default=300000,           help="N samples")
	parser.add_option("-l", "--latent",    dest="latent",      type="int",      default=100,              help="Dimension of latent space")
	(options, args) = parser.parse_args()
	
	# load model
	modelfile = options.model
	model = load_model(modelfile)
	# size of the latent space
	latent_dim = options.latent
	#Name of input file
	inputfile = options.input
	#Name of output file
	output = options.output
	#number of samples
	samples = options.samples
	
	scaler = StandardScaler()
	dataset = load_real_samples(inputfile, scaler)
	generate_and_save(model, dataset, latent_dim, samples, output, scaler)
