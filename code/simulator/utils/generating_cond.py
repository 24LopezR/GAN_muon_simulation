##################################################################################################
#### Sample generating module                                                                 ####
##################################################################################################
import numpy as np
from numpy import zeros
from numpy import ones
from numpy.random import randn, uniform
from numpy.random import randint

# # select real samples
def generate_real_samples(dataset, n_samples, weights=None, ind=1):
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
    if ind % 1 == 0:
        ix = randint(0, input_data.shape[0], n_samples)
    else:
        ix = randint(0, input_data.shape[0], n_samples)
    # select samples
    X, in_data = activations[ix], input_data[ix]
    if weights is not None:
        w = weights[ix]
    else:
        w = None
    # generate class labels
    y = ones((n_samples, 1))
    #print('Real samples')
    #print(X)
    return [X, in_data], w, y
 
# generate points in latent space as input for the generator
def generate_latent_points(dataset, latent_dim, n_samples, weights=None, ind=0):
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
    x_input = uniform(-1, 1, latent_dim * n_samples)
	# reshape into a batch of inputs for the network
    in_Z = x_input.reshape(n_samples, latent_dim)
	# choose random instances
    if ind % 1 == 0:
        ix = randint(0, input_data.shape[0], n_samples)
    else:
        ix = randint(0, input_data.shape[0], n_samples)
    in_data = input_data[ix]
    if weights is not None:
        w = weights[ix]
    else:
        w = None
    return [in_Z, in_data], w
 
# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, dataset, latent_dim, n_samples, weights=None, ind=0):
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
    [in_Z, in_data], w = generate_latent_points(dataset, latent_dim, n_samples, weights, ind=ind)
	# predict outputs
    X = generator.predict([in_Z, in_data])
    #print('Fake samples')
    #print(X)
    X = np.round(X)
    #print('Fake samples')
    #print((np.sum(X, axis=1)).size)
    #print((np.sum(X, axis=1)))
    #print(X[0,:])
	# create class labels
    y = zeros((n_samples, 1))
    return [X, in_data], w, y
	
# use the generator to generate n fake examples, with class labels
def generate_evaluation_samples(g_model, dataset, latent_dim, scaler):
    input_data, activations = dataset
	# generate points in the latent space
    n_samples = input_data.shape[0]
    print('> Number of evaluation samples = ', n_samples)
    in_Z = uniform(-1, 1, latent_dim * n_samples).reshape(n_samples, latent_dim)
	# predict outputs
    activations_gen = np.round((g_model.predict([in_Z, input_data])))
    #activations = scaler.inverse_transform(activations)
    return activations, activations_gen
