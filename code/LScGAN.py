############################################################################################
############################################################################################
### Generative Adversarial Network to simulate muon scattering                           ###
###   The input will be random noise and the measurements in the first detector          ###
#######                                                                                  ###
### Use: python cond_gan.py -i [input.root] -e [epochs] -l [latent dim] -k 5             ###
############################################################################################
############################################################################################
import numpy as np
import tensorflow as tf
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
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
from keras.layers import Activation
from matplotlib import pyplot
from optparse import OptionParser
import ROOT as r
from array import array
from sklearn.preprocessing import StandardScaler
from keras import backend
from keras.layers import BatchNormalization
from keras.initializers import RandomNormal
from keras.constraints import Constraint

# Wasserstein GAN implementation #########################################
WGAN = False
# implementation of wasserstein loss
def wasserstein_loss(y_true, y_pred):
	return backend.mean(y_true * y_pred)

# clip model weights to a given hypercube
class ClipConstraint(Constraint):
	# set clip value when initialized
	def __init__(self, clip_value):
		self.clip_value = clip_value
 
	# clip model weights to hypercube
	def __call__(self, weights):
		return backend.clip(weights, -self.clip_value, self.clip_value)
 
	# get the config
	def get_config(self):
		return {'clip_value': self.clip_value}
##########################################################################

# define the standalone discriminator model
def define_discriminator(in_shape=4):
	# first detector input
	in_first_det_data = Input(shape=in_shape)
	# second detector input
	in_second_det_data = Input(shape=in_shape)
	# concat label as a channel
	merge = Concatenate()([in_second_det_data, in_first_det_data])
	
	fe = Dense(128)(merge)
	fe = LeakyReLU(alpha=0.2)(fe)
	
	fe = Dense(64)(fe)
	fe = LeakyReLU(alpha=0.2)(fe)

	fe = Dense(64)(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	#fe = BatchNormalization()(fe)

	fe = Dense(64)(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	#fe = BatchNormalization()(fe)
	
	fe = Flatten()(fe)
	# output
	out_layer = Dense(1, activation='linear')(fe)
	# define model
	model = Model([in_second_det_data, in_first_det_data], out_layer)
	# compile model
	opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
	model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
	return model
 
# define the standalone generator model
def define_generator(latent_dim):
	# data input
	in_first_detector = Input(shape=4)
	# noise input
	in_lat = Input(shape=latent_dim)

	n_nodes = 4 * 127
	gen = Dense(n_nodes)(in_lat)

	merge = Concatenate()([gen, in_first_detector])
	gen = Dense(512)(merge)
	gen = Activation('relu')(gen)
	#gen = BatchNormalization()(gen)
	gen = Dense(256)(gen)
	gen = Activation('relu')(gen)
	#gen = BatchNormalization()(gen)
	gen = Dense(256)(gen)
	gen = Activation('relu')(gen)
	gen = Dense(128)(gen)
	gen = Activation('relu')(gen)
	gen = Dense(64)(gen)
	gen = Activation('relu')(gen)
	gen = Dense(16)(gen)
	gen = Activation('relu')(gen)
	# output
	out_layer = Dense(4, activation='linear')(gen)
	# define model
	model = Model([in_lat, in_first_detector], out_layer)
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
	# compile model
	opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
	model.compile(loss='mse', optimizer=opt)
	return model
 
# load muon data
def load_real_samples(inputfile, scaler):
	"""
	Reads the input file with the muon data and return the scaled dataset as a numpy array
	Arguments:
	Input:
		inputfile: name of the .root file with the data
		scaler: (sklearn.preprocessing.StandardScaler()) 
	Output: 
		dataset = [second_det, first_det]: numpy array of shape [(N, 4), (N, 4)]
		w: computed weights, numpy array of size (N, 1)
	"""
	f = r.TFile(inputfile)
	thedata = [] 
	for ev in f.globalReco:
            if ev.type1 != 3 or ev.type2 != 3:
                continue
            if abs(ev.px1) > 80 or abs(ev.py1) > 80 or abs(ev.pvx1) > 1.5 or abs(ev.pvy1) > 1.5:
                continue
            if abs(ev.px2) > 80 or abs(ev.py2) > 80 or abs(ev.pvx2) > 1.5 or abs(ev.pvy2) > 1.5:
                continue
            thedata.append([ev.px1, ev.py1, ev.pvx1, ev.pvy1, ev.px2-ev.px1 + 39*2 * ev.pvx1, ev.py2-ev.py1 + 39*2 * ev.pvy1, ev.pvx2-ev.pvx1, ev.pvy2-ev.pvy1])
	data = np.asarray(thedata)
	# weight events
	w = (data[:,4]**2+data[:,5]**2)
	scaler.fit(data, sample_weight = 1/w)
	data_transf = scaler.transform(data)
	first_det = data_transf[:,:4]
	second_det = data_transf[:,4:]
	
	# compute weights
	w = None
	return [second_det, first_det], w

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

def generate_and_save(g_model, dataset, latent_dim, n_samples, scaler, output):
	"""
	Generates fake samples and saves them in a .root file.
	Arguments:
	Input:
		g_mode: generator model
		dataset: set of real data with the structure from load_ral_samples()
		latent_dim
		n_samples: number of samples we want to generate
		scaler: StandardScaler() like
		output: name of output .root file
	Output:
		returns nothing
	"""
	f = r.TFile(output, "RECREATE")
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
	
def plot_history(d1_hist, d2_hist, g_hist):
	"""
	Plots the discriminator loss for real and fake samples and the generator loss, and saves the plot to a .png file.
	Arguments:
	Input:
		d1_hist: discriminator loss on real samples
		d2_hist: discriminator loss on fake samples
		g_hist: generator loss
	"""
	# plot history
	pyplot.plot(d1_hist, label='crit_real')
	pyplot.plot(d2_hist, label='crit_fake')
	pyplot.plot(g_hist, label='gen')
	pyplot.legend()
	pyplot.savefig('plot_line_plot_loss.png', dpi = 400)
	pyplot.close()

def save_model(epoch):
	# save the generator model tile file
	filename = 'generator_model_%03d.h5' % (epoch + 1)
	g_model.save(filename)

# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_discrim_updates, n_epochs=10, n_batch=128, weights=None):
	bat_per_epo = int(dataset[0].shape[0] / n_batch)
	half_batch = int(n_batch / 2)
	# lists for keeping track of loss (lari)
	# manually enumerate epochs
	d_real_hist, d_fake_hist, g_hist = list(), list(), list()
	for i in range(n_epochs):
		# enumerate batches over the training set
		for j in range(bat_per_epo):
			for k in range(n_discrim_updates):
				# get randomly selected 'real' samples
				[X_real, labels_real], w_real, y_real = generate_real_samples(dataset, half_batch, weights=weights)
				# update discriminator model weights
				d_loss1, acc_real = d_model.train_on_batch([X_real, labels_real], y_real, sample_weight=w_real)
				# generate 'fake' examples
				[X_fake, labels], w_fake, y_fake = generate_fake_samples(g_model, dataset, latent_dim, half_batch, weights=weights)
				# update discriminator model weights
				d_loss2, acc_fake = d_model.train_on_batch([X_fake, labels], y_fake, sample_weight=w_fake)
			# prepare points in latent space as input for the generator
			[z_input, labels_input], w_input = generate_latent_points(dataset, latent_dim, n_batch, weights=weights)
			# create inverted labels for the fake samples
			y_gan = ones((n_batch, 1))
			# update the generator via the discriminator's error
			g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan, sample_weight=w_input)
			if (j+1) % 20 == 0:
				print('>%d, %d/%d, d1=%.3f, d2=%.3f, g=%.3f; Accuracy real: %.0f%%, fake: %.0f%%' % (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss, acc_real*100, acc_fake*100))
				d_real_hist.append(d_loss1)
				d_fake_hist.append(d_loss2)
				g_hist.append(g_loss)
		plot_history(d_real_hist,d_fake_hist,g_hist)
		# evaluate the model performance, sometimes
		if (i+1) % 10 == 0:
			save_model(i)
	# save the generator model
	g_model.save('cgan_generator.h5')

##################################################################
##################################################################
########################### Main #################################
##################################################################
##################################################################
if __name__ == "__main__":

	parser = OptionParser(usage="%prog --help")
	parser.add_option("-i", "--input",     dest="input",       type="string",   default='input.root',     help="Input root file")
	parser.add_option("-o", "--output",    dest="output",      type="string",   default='output.root',    help="Output root file")
	parser.add_option("-e", "--epochs",    dest="epochs",      type="int",      default=100,              help="N epochs")
	parser.add_option("-l", "--latent",    dest="latent",      type="int",      default=64,               help="Dimension of latent space")
	parser.add_option("-b", "--batch",     dest="batch",       type="int",      default=256,              help="N batch")
	parser.add_option("-k", "--khyp",      dest="k_hyp",       type="int",      default=5,                help="k hyperparameter")
	(options, args) = parser.parse_args()
	
	config = tf.compat.v1.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.compat.v1.Session(config=config)
	backend.set_session(sess)

	# size of the latent space
	latent_dim = options.latent
	#Name of input file
	inputfile = options.input
	#Name of output file
	output = options.output
	#number of batch
	nbatch = options.batch
	#number of batch
	epochs = options.epochs
	#number or discriminator updates
	n_discrim_updates = options.k_hyp
	# create the discriminator
	d_model = define_discriminator()
	d_model.summary()
	# create the generator
	g_model = define_generator(latent_dim)
	g_model.summary()
	# create the gan
	gan_model = define_gan(g_model, d_model)
	gan_model.summary()

	# load image data
	scaler = StandardScaler()
	dataset, weights = load_real_samples(inputfile, scaler)
	# train model
	train(g_model, d_model, gan_model, dataset, latent_dim, n_discrim_updates, epochs, nbatch, weights)
	#Generate a few
	generate_and_save(g_model, dataset, latent_dim, 300000, scaler, output)
