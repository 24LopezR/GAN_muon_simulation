import ROOT as r
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages
import sys
from optparse import OptionParser
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
from array import array
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from numpy import genfromtxt
from os import listdir
from os.path import isfile, isdir, join
from scipy.stats import skew, kstest
from numpy import format_float_scientific as scifor

# load muon data
def load_real_samples(inputfile, scaler):
	"""
	Reads the input file with the muon data and return the scaled dataset as a numpy array
	Arguments:
	Input:
		inputfile: name of the .root file with the data
		scaler: (sklearn.preprocessing.StandardScaler()) 
	Output: 
		data_transf: numpy array of dimension (n_events, 8) with scaled data. 4 first variables correspond to first detector data
			and 4 last variables to second detector data
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
	return [second_det, first_det]

# use the generator to generate n fake examples, with class labels
def generate_evaluation_samples(generator, dataset, latent_dim):
	# generate points in latent space
	second_detector, first_detector = dataset
	# generate points in the latent space
	n_samples = first_detector.shape[0]
	print('> N of evaluation samples = ', n_samples)
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	# choose random instances
	ix = np.arange(n_samples)
	input_data = first_detector[ix]
	real = second_detector[ix]	
	# predict outputs
	fake = generator.predict([z_input, input_data])
	return real, fake

def compute_distance(real, fake):
	distance = 0
	for i in np.arange(real.shape[0]):
		d_temp = real[i,:]-fake[i,:]
		distance = distance + np.dot(d_temp, d_temp)
	return distance / real.shape[0]
	
def get_mean_difference(real, fake):
	means_real = np.mean(real, axis=0)
	means_fake = np.mean(fake, axis=0)
	err_real = np.std(real, axis=0, ddof=1)/np.sqrt(real.shape[0])
	err_fake = np.std(fake, axis=0, ddof=1)/np.sqrt(fake.shape[0])
	pull = (means_real - means_fake) / np.sqrt(err_real**2 + err_fake**2)
	return pull

def get_cov_matrices(real, fake):
	real_cov = np.cov(real, rowvar=False)
	fake_cov = np.cov(fake, rowvar=False)
	return real_cov, fake_cov, real_cov-fake_cov

def get_skewness(real, fake):
	skew_real = skew(real)
	skew_fake = skew(fake)
	return skew_real, skew_fake, skew_real-skew_fake

def ks_test(real, fake):
	p_values = [kstest(real[:,0], fake[:,0])[1], 
	            kstest(real[:,1], fake[:,1])[1], 
	            kstest(real[:,0], fake[:,2])[1], 
	            kstest(real[:,1], fake[:,3])[1]]
	return p_values
	
def print_results(modelname, pull, cov, skew, p_values):
	cov_real, cov_fake, cov_dif = cov
	skew_real, skew_fake, skew_dif = skew
	
	print("."*90)
	print("    Summary of results: {}".format('model.h5'))
	print("."*90)
	print("{:<20} {:<15} {:<15} {:<15} {:<15}".format('Parameter','Dx','Dy','Dv_x','Dv_y'))
	print("."*90)
	print("{:<20} {:<15} {:<15} {:<15} {:<15}".format('Pull',
		                                         scifor(pull[0],precision=3),
		                                         scifor(pull[1],precision=3),
		                                         scifor(pull[2],precision=3),
		                                         scifor(pull[3],precision=3)))
	print(" "*90)
	print("{:<20} {:<15} {:<15} {:<15} {:<15}".format('Skewness_real',
		                                         scifor(skew_real[0],precision=3),
		                                         scifor(skew_real[1],precision=3),
		                                         scifor(skew_real[2],precision=3),
		                                         scifor(skew_real[3],precision=3)))
	print("{:<20} {:<15} {:<15} {:<15} {:<15}".format('Skewness_fake',
		                                         scifor(skew_fake[0],precision=3),
		                                         scifor(skew_fake[1],precision=3),
		                                         scifor(skew_fake[2],precision=3),
		                                         scifor(skew_fake[3],precision=3)))
	print(" "*90)
	print("{:<20} {:<15} {:<15} {:<15} {:<15}".format('KS-test (p-value)',
		                                         scifor(p_values[0],precision=3),
		                                         scifor(p_values[1],precision=3),
		                                         scifor(p_values[2],precision=3),
		                                         scifor(p_values[3],precision=3)))
	print("."*90)
	print("    Covariance matrices")
	print("."*90)
	print("Real samples:")
	print("")
	print('\n'.join([''.join(['{:<10.2f}'.format(item) for item in row]) 
	      for row in cov_real]))
	print("."*90)
	print("Fake samples:")
	print("")
	print('\n'.join([''.join(['{:<10.2f}'.format(item) for item in row]) 
	      for row in cov_fake]))
	     
#####################################################################################################################################################
##########################  Plotting functions ######################################################################################################
#####################################################################################################################################################

def plot_vector_difference(scale):
    plt.rcParams["figure.figsize"] = (14,7)

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    xrange = (-0.3,0.3)

    ax1.hist(real[:,2], bins=100, density = True, range = xrange, histtype = 'step', color = 'black', log = False, label = 'data $\Delta v_x$')
    ax1.hist(fake[:,2], bins=100, density = True, range = xrange, histtype = 'step', color = 'red', log = False, label = 'generated $\Delta v_x$')
    ax1.set_xlabel('$\Delta v_x$')
    ax1.set_yscale(scale)
    ax1.legend()
    ax2.hist(real[:,3], bins=100, density = True, range = xrange, histtype = 'step', color = 'black', log = False, label = 'data $\Delta v_y$')
    ax2.hist(fake[:,3], bins=100, density = True, range = xrange, histtype = 'step', color = 'red', log = False, label = 'generated $\Delta v_x$')
    ax2.set_xlabel('$\Delta v_y$')
    ax2.set_yscale(scale)
    ax2.legend()
    fig.suptitle('Vector difference');
    return fig

def plot_position_difference(scale):
    plt.rcParams["figure.figsize"] = (14,7)
    xrange = (-5,5)

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)

    h1 = ax1.hist(real[:,0], bins=100, density = True, range = (-5,5), histtype = 'step', color = 'black', label='data $\Delta x$')
    h2 = ax2.hist(real[:,1], bins=100, density = True, range = (-5,5), histtype = 'step', color = 'black', label='data $\Delta y$')
    if False:
        popt, pcov = curve_fit(gauss_function, np.linspace(*xrange, 100), h1[0])
        ax1.plot(np.linspace(*xrange, 100), gauss_function(np.linspace(*xrange, 100), *popt), label=f'gaussian fit: \n [{popt[0]:0.2e}, \n {popt[1]:0.2e}, \n {popt[2]:0.2e}]', color = 'black')
        popt, pcov = curve_fit(gauss_function, np.linspace(*xrange, 100), h2[0])
        ax2.plot(np.linspace(*xrange, 100), gauss_function(np.linspace(*xrange, 100), *popt), label=f'gaussian fit: \n [{popt[0]:0.2e}, \n {popt[1]:0.2e}, \n {popt[2]:0.2e}]', color = 'red')
    h1 = ax1.hist(fake[:,0], bins=100, density = True, range = (-5,5), histtype = 'step', color = 'red', label='generated $\Delta x$')
    h2 = ax2.hist(fake[:,1], bins=100, density = True, range = (-5,5), histtype = 'step', color = 'red', label='generated $\Delta y$')
    ax1.set_xlabel('$\Delta x$')
    ax2.set_xlabel('$\Delta y$')
    ax1.set_yscale(scale)
    ax2.set_yscale(scale)
    ax1.legend()
    ax2.legend()

    fig.suptitle('Position difference');
    return fig

def plot_correlation_2Dhist(var1, var2, var1g, var2g, ranges, labels):
    plt.rcParams["figure.figsize"] = (14,7)

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)

    ax1.hist2d(var1, var2, bins=100, range = ranges, density = True,
              norm=matplotlib.colors.LogNorm())
    ax1.set_xlabel(labels[0])
    ax1.set_ylabel(labels[1])
    ax2.hist2d(var1g, var2g, bins=100, range = ranges, density = True,
              norm=matplotlib.colors.LogNorm())
    ax2.set_xlabel(labels[0])
    ax2.set_ylabel(labels[1])
    fig.suptitle('Correlation '+labels[0]+'-'+labels[1]);
    return fig

def plot_history(d1_hist, d2_hist, g_hist, y1, y2):
        # plot history
        pyplot.plot(d1_hist, label='crit_real')
        pyplot.plot(d2_hist, label='crit_fake')
        pyplot.plot(g_hist, label='gen')
        pyplot.legend()
        pyplot.ylim([y1,y2])
        pyplot.savefig('plot_line_plot_loss.png', dpi = 400)
        pyplot.close()

##################################################################
##################################################################
########################### Main #################################
##################################################################
##################################################################
if __name__ == "__main__":

	parser = OptionParser(usage="%prog --help")
	parser.add_option("-i", "--input",     dest="input",       type="string",   default='input.root',     help="Input root file")
	parser.add_option("-m", "--models",     dest="models",       type="string",   default='./models',      help="Models folder / model .h5 file")
	parser.add_option("-c", "--csv",       dest="loss",       type="string",   default=None,       help="Loss data file")
	parser.add_option("-l", "--latent",    dest="latent",      type="int",      default=100,              help="Dimension of latent space")
	parser.add_option('-q', '--quality', action='store_true', dest='mode')
	(options, args) = parser.parse_args()
		
	# size of the latent space
	latent_dim = options.latent
	#Name of input file
	inputfile = options.input
	# Loss data
	loss_file = options.loss
	
	print_q = options.mode
	
	if print_q:
		if isdir(options.models):
			# load model
			models_dir = options.models
			# load data once 
			scaler = StandardScaler()
			dataset = load_real_samples(inputfile, scaler)
			
			# calculate metric for all models in directory
			epochs = []
			q_hist = []
			models = [f for f in listdir(models_dir) if isfile(join(models_dir, f))]
			print(models)
			for modelfile in models:
				epoch = int(modelfile.replace('generator_model_','').replace('.h5',''))
				print('-----------------------------------------------------------------')
				print('> epoch: '+str(epoch))
				epochs.append(epoch)
				model = load_model(models_dir+'/'+modelfile)
				real, fake = generate_evaluation_samples(model, dataset, latent_dim)
				distance = compute_distance(real, fake)
				print('> model: '+modelfile+' | D = %.3f' % distance)
				q_hist.append(distance)
				print(epochs)
				print(q_hist)
			
			# plot q_hist
			epochs = np.array(epochs)
			q_hist = np.array(q_hist)
			o = np.argsort(epochs)
			epochs = epochs[o]
			q_hist = q_hist[o]
			pyplot.plot(epochs, q_hist, label='Q')
			pyplot.xlabel('epoch')
			pyplot.ylabel('Q')
			pyplot.savefig(models_dir+'/metrics.png', dpi = 400)
			pyplot.close()
		else:
			modelfile = options.models
			model = load_model(modelfile)
			# load data once 
			scaler = StandardScaler()
			dataset = load_real_samples(inputfile, scaler)
			# generate real and fake samples
			real, fake = generate_evaluation_samples(model, dataset, latent_dim)
			
			pull = get_mean_difference(real, fake)
			cov = get_cov_matrices(real,fake)
			skew = get_skewness(real, fake)
			p_values = ks_test(real, fake)
			
			print_results(modelfile, pull, cov, skew, p_values)
			
			out = PdfPages('evaluation.pdf')
			out.savefig(plot_vector_difference('log'));
			out.savefig(plot_position_difference('log'));
			out.savefig(plot_correlation_2Dhist(real[:,0], real[:,2], fake[:,0], fake[:,2], [[-5,5],[-0.5,0.5]], ['$\Delta x$','$\Delta v_x$']));
			out.savefig(plot_correlation_2Dhist(real[:,1], real[:,3], fake[:,1], fake[:,3], [[-5,5],[-0.5,0.5]], ['$\Delta y$','$\Delta v_y$']));
			out.close()
			
			
			
	
	# plot loss
	if loss_file:
		loss_data = genfromtxt(loss_file, delimiter=' ')
		plot_history(loss_data[:,1], loss_data[:,2], loss_data[:,3], 0.5, 1)
	
