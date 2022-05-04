##################################################################################################
#### Sample generating module                                                                 ####
##################################################################################################
import numpy as np
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint, uniform
import utils.cfg as cfg

SIGMA = 1e-3

def get_activations(X):
    X = X.argmax(axis=2)
    strings = [''.join([np.base_repr(i, 2).zfill(4) for i in X[j]]) for j in range(X.shape[0])]
    activations = [np.asarray(list(s), dtype=int) for s in strings]
    return activations

def generate_labels(value, dim, threshold=1):
    # generate class labels
    y = zeros(dim) if value == 0 else ones(dim)
    z = uniform(0, 1, dim)
    mask = z > threshold
    y = np.logical_xor(y, mask)
    return 1*y

# # select real samples
def generate_real_samples(dataset, n_samples):
    '''
    Generates a set of samples from the real data to train the discriminator.

    Parameters
    ----------
    dataset : TYPE
        DESCRIPTION.
    n_samples : TYPE
        DESCRIPTION.
    weights : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    [X, in_data] :
        Random chosen real samples.
    w : NumPy Array
        Weights.
    y : NumPy Array
        Class labels.

    '''
    input_data, activations = dataset
    # choose random instances
    ix = randint(0, input_data.shape[0], n_samples)
    # select samples
    X, in_data = activations[ix], input_data[ix]
    # add noise to generated samples
    #print(X)
    X = X + np.reshape(SIGMA * randn(X.size), X.shape)
    # generate class labels
    y = generate_labels(1, (n_samples, 1), threshold=1)
    #print('Real samples')
    #print(X)
    return [X, in_data], y
 
# generate points in latent space as input for the generator
def generate_latent_points(dataset, latent_dim, n_samples):
    '''
    Generates points in the latent space.

    Parameters
    ----------
    dataset : TYPE
        DESCRIPTION.
    latent_dim : TYPE
        DESCRIPTION.
    n_samples : TYPE
        DESCRIPTION.
    weights : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    [in_Z, in_data] :
        Generated latent points.
    w : NumPy Array
        Weights.

    '''
    input_data, activations = dataset
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
    in_Z = x_input.reshape(n_samples, latent_dim)
	# choose random instances
    ix = randint(0, input_data.shape[0], n_samples)
    in_data = input_data[ix]
    return [in_Z, in_data]
 
# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, dataset, latent_dim, n_samples, print_x=False):
    '''
    Generates a set of samples from a keras model to train the discriminator.

    Parameters
    ----------
    generator : TYPE
        DESCRIPTION.
    dataset : TYPE
        DESCRIPTION.
    latent_dim : TYPE
        DESCRIPTION.
    n_samples : TYPE
        DESCRIPTION.
    weights : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    [X, in_data] :
        Generated samples.
    w : NumPy Array
        Weights.
    y : NumPy Array
        Class labels.

    '''
	# generate points in latent space
    [in_Z, in_data] = generate_latent_points(dataset, latent_dim, n_samples)
	# predict outputs
    if cfg.CONDITIONAL:
        X = generator.predict([in_Z, in_data])
    else:
        X = generator.predict(in_Z)
    if print_x:
        # number of hits
        #print(X.shape)
        print(np.unique(np.sum(np.argmax(X, axis=-1), axis=1)))
        # unique values
        #act = np.argmax(X, axis=1)
        #print('Unique samples: {:<5}'.format(np.unique(act).size))
    # add noise to generated samples
    X = X + np.reshape(SIGMA * randn(X.size), X.shape)
	# create class labels
    y = generate_labels(0, (n_samples, 1), threshold=1)
    return [X, in_data], y
	
# use the generator to generate n fake examples, with class labels
def generate_evaluation_samples(g_model, dataset, latent_dim):
    input_data, activations = dataset
	# generate points in the latent space
    n_samples = input_data.shape[0]
    print('> Number of evaluation samples = ', n_samples)
    in_Z = randn(latent_dim * n_samples).reshape(n_samples, latent_dim)
	# predict outputs
    if cfg.CONDITIONAL:
        activations_gen = g_model.predict([in_Z, input_data])
    else:
        activations_gen = g_model.predict(in_Z)
    # scale back the results
    activations     = np.argmax(activations, axis=1)
    activations_gen = np.argmax(activations_gen, axis=1)
    return activations, activations_gen
