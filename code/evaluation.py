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
from plot_results import plot_difference, plot_correlation_2Dhist

# use the generator to generate n fake examples, with class labels
def generate_evaluation_samples(generator, inputfile, latent_dim, scaler):
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
	first_det = data[:,:4]
	second_det = data[:,4:]
	real = second_det
	
	# scale real events
	w = 1/np.sqrt(data[:,4]**2 + data[:,5]**2)# + data[:,6]**2 + data[:,7]**2)
	scaler.fit(data, sample_weight=w)
	real_data_transf = scaler.transform(data)
	# generate points in the latent space
	n_samples = first_det.shape[0]
	print('> N of evaluation samples = ', n_samples)
	noise_input = randn(latent_dim * n_samples).reshape(n_samples, latent_dim)
	data_input = real_data_transf[:,:4] # take first detector info
	# predict outputs
	raw_predictions = generator.predict([noise_input, data_input])
	# scale back the generated events
	gen_data = np.hstack((data_input, raw_predictions))
	fake = scaler.inverse_transform(gen_data)[:,4:]
	return real, fake
	
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
	p=5
	
	print("."*90)
	print("    Summary of results: {}".format(modelname))
	print("."*90)
	print("{:<20} {:<15} {:<15} {:<15} {:<15}".format('Parameter','Dx','Dy','Dv_x','Dv_y'))
	print("."*90)
	print("{:<20} {:<15} {:<15} {:<15} {:<15}".format('Pull',
		                                         scifor(pull[0],precision=p),
		                                         scifor(pull[1],precision=p),
		                                         scifor(pull[2],precision=p),
		                                         scifor(pull[3],precision=p)))
	print(" "*90)
	print("{:<20} {:<15} {:<15} {:<15} {:<15}".format('Skewness_real',
		                                         scifor(skew_real[0],precision=p),
		                                         scifor(skew_real[1],precision=p),
		                                         scifor(skew_real[2],precision=p),
		                                         scifor(skew_real[3],precision=p)))
	print("{:<20} {:<15} {:<15} {:<15} {:<15}".format('Skewness_fake',
		                                         scifor(skew_fake[0],precision=p),
		                                         scifor(skew_fake[1],precision=p),
		                                         scifor(skew_fake[2],precision=p),
		                                         scifor(skew_fake[3],precision=p)))
	print(" "*90)
	print("{:<20} {:<15} {:<15} {:<15} {:<15}".format('KS-test (p-value)',
		                                         scifor(p_values[0],precision=p),
		                                         scifor(p_values[1],precision=p),
		                                         scifor(p_values[2],precision=p),
		                                         scifor(p_values[3],precision=p)))
	print("."*90)
	print("    Covariance matrices")
	print("."*90)
	print("Real samples:")
	print("")
	print('\n'.join([''.join(['{:<12.7f}'.format(item) for item in row]) 
	      for row in cov_real]))
	print("."*90)
	print("Fake samples:")
	print("")
	print('\n'.join([''.join(['{:<12.7f}'.format(item) for item in row]) 
	      for row in cov_fake]))

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
	parser.add_option("-i", "--input",     dest="input",       type="string",   default='input.root',         help="Input root file")
	parser.add_option("-o", "--output",    dest='output',      type="string",   default='evaluation.pdf',     help='Output filename')
	parser.add_option("-m", "--model",     dest="model",       type="string",   default=None,                 help="Model .h5 file")
	parser.add_option("-c", "--csv",       dest="loss",        type="string",   default=None,                 help="Loss data file")
	parser.add_option("-l", "--latent",    dest="latent",      type="int",      default=64,                   help="Dimension of latent space")
	(options, args) = parser.parse_args()

	# Name of input file
	input_file = options.input
	# Name of output file
	output_file = options.output
	# Model file
	model_file = options.model
	# Loss data
	loss_file = options.loss
	# size of the latent space
	latent_dim = options.latent

	# Evaluate the model
	if model_file:
		# Load the model
		model = load_model(model_file)
		# generate real and fake samples
		scaler = StandardScaler()
		real, fake = generate_evaluation_samples(model, input_file, latent_dim, scaler)

		pull = get_mean_difference(real, fake)
		cov = get_cov_matrices(real,fake)
		skew = get_skewness(real, fake)
		p_values = ks_test(real, fake)

		print_results(model_file, pull, cov, skew, p_values)

		out = PdfPages(output_file)
		out.savefig(plot_difference(real[:,0], real[:,1], fake[:,0], fake[:,1], (-50,50), ['$\Delta x$','$\Delta y$'], 'log'));
		out.savefig(plot_difference(real[:,2], real[:,3], fake[:,2], fake[:,3], (-1.5,1.5), ['$\Delta v_x$','$\Delta v_y$'], 'log'));
		out.savefig(plot_correlation_2Dhist(real[:,0], real[:,2], fake[:,0], fake[:,2], [[-15,15],[-1,1]], ['$\Delta x$','$\Delta v_x$']));
		out.savefig(plot_correlation_2Dhist(real[:,1], real[:,3], fake[:,1], fake[:,3], [[-15,15],[-1,1]], ['$\Delta y$','$\Delta v_y$']));
		out.close()

	# plot loss
	if loss_file:
		loss_data = genfromtxt(loss_file, delimiter=' ')
		plot_history(loss_data[:,1], loss_data[:,2], loss_data[:,3], 0.5, 1)

