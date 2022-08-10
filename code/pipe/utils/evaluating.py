##################################################################################################
#### Evaluation module                                                                        ####
##################################################################################################
import numpy as np
from scipy.stats import skew, kstest
from numpy import format_float_scientific as scifor
	
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
	return real_cov, fake_cov

def get_skewness(real, fake):
	skew_real = skew(real)
	skew_fake = skew(fake)
	return skew_real, skew_fake

def ks_test(real, fake):
	p_values = [kstest(real[:,0], fake[:,0])[1], 
	            kstest(real[:,1], fake[:,1])[1], 
	            kstest(real[:,2], fake[:,2])[1],
	            kstest(real[:,3], fake[:,3])[1]]
	return p_values
	
def print_results(pull, cov, skew, p_values):
	cov_real, cov_fake, cov_dif = cov
	skew_real, skew_fake, skew_dif = skew
	p=5
	
	print("."*90)
	print("    Summary of results")
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
	      
def evaluate(real, fake):
	#pull = get_mean_difference(real, fake)
	#cov = get_cov_matrices(real,fake)
	#skew = get_skewness(real, fake)
	#p_values = ks_test(real, fake)
	#print_results(pull, cov, skew, p_values)

	# Calculate means
	means_real = np.mean(real, axis=0)
	means_fake = np.mean(fake, axis=0)

	# Calculate skewness
	skew_real, skew_fake = get_skewness(real, fake)

	# Calculate covariance matrices
	cov_real, cov_fake = get_cov_matrices(real, fake)

	print("." * 90)
	print("    Summary of results")
	print("." * 90)
	print("{:<20} {:<15} {:<15} {:<15} {:<15}".format('Parameter', 'Dx', 'Dy', 'Dv_x', 'Dv_y'))
	print("." * 90)
	print("{:<20} {:<15.7e} {:<15.7e} {:<15.7e} {:<15.7f}".format('Mean real',means_real[0],means_real[1], means_real[2], means_real[3]))
	print("{:<20} {:<15.7e} {:<15.7e} {:<15.7e} {:<15.7f}".format('Mean gen', means_fake[0],means_fake[1], means_fake[2], means_fake[3]))
	print("{:<20} {:<15.7f} {:<15.7f} {:<15.7f} {:<15.7f}".format('Skew real', skew_real[0], skew_real[1], skew_real[2], skew_real[3]))
	print("{:<20} {:<15.7f} {:<15.7f} {:<15.7f} {:<15.7f}".format('Skew gen', skew_fake[0], skew_fake[1], skew_fake[2], skew_fake[3]))
	print("." * 90)
	print("    Covariance matrices")
	print("." * 90)
	print("Real samples:")
	print("")
	print('\n'.join([''.join(['{:<12.7f}'.format(item) for item in row])
					 for row in cov_real]))
	print("." * 90)
	print("Fake samples:")
	print("")
	print('\n'.join([''.join(['{:<12.7f}'.format(item) for item in row])
					 for row in cov_fake]))


