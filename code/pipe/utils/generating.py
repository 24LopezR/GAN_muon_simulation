##################################################################################################
#### Sample generating module                                                                 ####
##################################################################################################
import numpy as np
import tensorflow as tf
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
import ROOT as r
from array import array

# # select real samples
def generate_real_samples(dataset, n_samples, weights=None):
	"""
	Generates a set of samples from the real data to train the discriminator.
	Arguments:
	Input:
		dataset: numpy array of shape [(N, 4), (N, 4)] (output from load_real_samples())
		n_samples: number of samples to generate
		weights: numpy array of size (N, 1) (output from load_real_samples())
	Output:
		X: real samples, numpy array of shape (N, 4)
		input_data: input variables corresponding to samples from X, numpy array of shape (N, 4)
		w: weights corresponding to samples from X, numpy array of shape (N, 1)
		y: class labels, numpy array of shape (N, 1)
	"""
	# split into first and second detector information
	# the info of first detector will be used as input to the deiscriminator and generator
	second_detector, first_detector = dataset
	# choose random instances
	ix = randint(0, second_detector.shape[0], n_samples)
	# select samples
	X, input_data = second_detector[ix], first_detector[ix]
	if weights is not None:
		w = weights[ix]
	else:
		w = None
	# generate class labels
	y = ones((n_samples, 1))
	return [X, input_data], w, y
 
# generate points in latent space as input for the generator
def generate_latent_points(dataset, latent_dim, n_samples, weights=None):
	"""
	Generates a set of samples from the real data to train the discriminator.
	Arguments:
	Input:
		dataset: numpy array of shape [(N, 4), (N, 4)] (output from load_real_samples())
		latent_dim: dimension of latent space
		n_samples: number of samples to generate
		weights: numpy array of size (N, 1) (output from load_real_samples())
	Output:
		z_input: latent space poin, numpy array of shape (N, 4)
		input_data: input variables corresponding to samples from X, numpy array of shape (N, 4)
		w: weights corresponding to samples from X, numpy array of shape (N, 1)
		y: class labels, numpy array of shape (N, 1)
	"""
	second_detector, first_detector = dataset
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	# choose random instances
	ix = randint(0, first_detector.shape[0], n_samples)
	input_data = first_detector[ix]
	if weights is not None:
		w = weights[ix]
	else:
		w = None
	return [z_input, input_data], w
 
# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, dataset, latent_dim, n_samples, weights=None):
	"""
	Generates a set of samples from a keras model to train the discriminator.
	Arguments:
	Input:
		generator: generator keras model
		dataset: numpy array of shape [(N, 4), (N, 4)] (output from load_real_samples())
		latent_dim: dimension of latent space
		n_samples: number of samples to generate
		weights: numpy array of size (N, 1) (output from load_real_samples())
	Output:
		X: fake samples, numpy array of shape (N, 4)
		input_data: input variables corresponding to samples from X, numpy array of shape (N, 4)
		w: weights corresponding to samples from X, numpy array of shape (N, 1)
		y: class labels, numpy array of shape (N, 1)
	"""
	# generate points in latent space
	[z_input, input_data], w = generate_latent_points(dataset, latent_dim, n_samples, weights)
	# predict outputs
	X = generator.predict([z_input, input_data])
	# create class labels
	y = zeros((n_samples, 1))
	return [X, input_data], w, y
	
# use the generator to generate n fake examples, with class labels
def generate_evaluation_samples(g_model, dataset, latent_dim, scaler):
	[second_det, first_det] = dataset
	# generate points in the latent space
	n_samples = first_det.shape[0]
	print('> Number of evaluation samples = ', n_samples)
	noise_input = randn(latent_dim * n_samples).reshape(n_samples, latent_dim)
	# predict outputs
	raw_predictions = g_model.predict([noise_input, first_det])
	# scale back the generated events
	gen_raw_data = np.hstack((first_det, raw_predictions))
	fake = scaler.inverse_transform(gen_raw_data)[:,4:]
	return fake

def generate_and_save(g_model, dataset, latent_dim, n_samples, scaler, output):
	"""
	Generates fake samples and saves them in a .root file.
	Arguments:
	Input:
		g_mode: generator model
		dataset: set of real data with the structure from load_real_samples()
		latent_dim
		n_samples: number of samples we want to generate
		scaler: StandardScaler() like
		output: name of output .root file
	Output:
		returns nothing
	"""
	f = r.TFile(output+'.root', "RECREATE")
	tree = r.TTree("globalReco", "globalReco")
	px1 = array('f', [0.])
	py1 = array('f', [0.])
	pvx1 = array('f', [0.])
	pvy1 = array('f', [0.])
	px2 = array('f', [0.])
	py2 = array('f', [0.])
	pvx2 = array('f', [0.])
	pvy2 = array('f', [0.])
	tree.Branch("px1", px1, 'px1/F')
	tree.Branch("py1", py1, 'py1/F')
	tree.Branch("pvx1", pvx1, 'pvx1/F')
	tree.Branch("pvy1", pvy1, 'pvy1/F')
	tree.Branch("px2", px2, 'px2/F')
	tree.Branch("py2", py2, 'py2/F')
	tree.Branch("pvx2", pvx2, 'pvx2/F')
	tree.Branch("pvy2", pvy2, 'pvy2/F')
	[Xtrans, input_data], _, _ = generate_fake_samples(g_model, dataset, latent_dim, n_samples)
	data = np.hstack((input_data, Xtrans))
	G = scaler.inverse_transform(data)
	for i in range(G.shape[0]):
		px1[0] = G[i,0]
		py1[0] = G[i,1]
		pvx1[0] = G[i,2]
		pvy1[0] = G[i,3]
		px2[0] = G[i,4] + px1[0] - 39*2 * pvx1[0]
		py2[0] = G[i,5] + py1[0] - 39*2 * pvy1[0]
		pvx2[0] = G[i,6] + pvx1[0]
		pvy2[0] = G[i,7] + pvy1[0]
		tree.Fill()
	tree.Write()
	f.Close()
