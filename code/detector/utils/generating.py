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
import utils.config as config

def nearest_point(X):
    X1 = []
    for i in range(config.INPUT_DIM):
        X1.append([find_nearest(config.DATASET_UNIQUES[i], X[j,i]) for j in range(X.shape[0])])
    return np.asarray(X1).T

def find_nearest(array, value):
    """
    Parameters
    ----------
    array : NumPy Array
        Array of values to search from
    value : TYPE
        Input value of which we want the closest one in 'array'

    Returns
    -------
    Numerical value in 'array' which is closest to 'value'
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

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
	m_vars, p_vars = dataset
	# choose random instances
	ix = randint(0, m_vars.shape[0], n_samples)
	# select samples
	X, input_data = m_vars[ix], p_vars[ix]
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
	m_vars, p_vars = dataset
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	# choose random instances
	ix = randint(0, p_vars.shape[0], n_samples)
	input_data = p_vars[ix]
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
    X = nearest_point(X)
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
    raw_predictions = nearest_point(raw_predictions)
	# scale back the generated events
    gen_raw_data = np.hstack((first_det, raw_predictions))
    fake = scaler.inverse_transform(gen_raw_data)[:,4:]
    return fake
