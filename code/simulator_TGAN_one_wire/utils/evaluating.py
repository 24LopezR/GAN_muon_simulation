##################################################################################################
#### Evaluation module                                                                        ####
##################################################################################################
import numpy as np
from scipy.stats import skew, kstest, wasserstein_distance
from numpy import format_float_scientific as scifor

MEAN_R        = 0.
MEAN_F        = 0.
STD_R         = 0.
STD_F         = 0.
SKEW_R        = 0.
SKEW_F        = 0.
KS            = 0.
WASSERSTEIN_D = 0.
PULL          = 0.
	
def get_mean_difference(real, fake):
    means_real = np.mean(real, axis=0)
    means_fake = np.mean(fake, axis=0)
    err_real = np.std(real, axis=0, ddof=1)/np.sqrt(real.shape[0])
    err_fake = np.std(fake, axis=0, ddof=1)/np.sqrt(fake.shape[0])
    pull = (means_real - means_fake) / np.sqrt(err_real**2 + err_fake**2)
    return means_real, means_fake, np.std(real), np.std(fake), pull

def get_cov_matrices(real, fake):
	real_cov = np.cov(real, rowvar=False)
	fake_cov = np.cov(fake, rowvar=False)
	return real_cov, fake_cov, real_cov-fake_cov

def get_skewness(real, fake):
	return skew(real), skew(fake)

def ks_test(real, fake):
	return kstest(real, fake)[1]
	
# def print_results(pull, cov, skew, p_values):
# 	cov_real, cov_fake, cov_dif = cov
# 	skew_real, skew_fake, skew_dif = skew
# 	p=5
# 	
# 	print("."*90)
# 	print("    Summary of results")
# 	print("."*90)
# 	print("{:<20} {:<15} {:<15} {:<15} {:<15}".format('Parameter','Dx','Dy','Dv_x','Dv_y'))
# 	print("."*90)
# 	print("{:<20} {:<15} {:<15} {:<15} {:<15}".format('Pull',
# 		                                         scifor(pull[0],precision=p),
# 		                                         scifor(pull[1],precision=p),
# 		                                         scifor(pull[2],precision=p),
# 		                                         scifor(pull[3],precision=p)))
# 	print(" "*90)
# 	print("{:<20} {:<15} {:<15} {:<15} {:<15}".format('Skewness_real',
# 		                                         scifor(skew_real[0],precision=p),
# 		                                         scifor(skew_real[1],precision=p),
# 		                                         scifor(skew_real[2],precision=p),
# 		                                         scifor(skew_real[3],precision=p)))
# 	print("{:<20} {:<15} {:<15} {:<15} {:<15}".format('Skewness_fake',
# 		                                         scifor(skew_fake[0],precision=p),
# 		                                         scifor(skew_fake[1],precision=p),
# 		                                         scifor(skew_fake[2],precision=p),
# 		                                         scifor(skew_fake[3],precision=p)))
# 	print(" "*90)
# 	print("{:<20} {:<15} {:<15} {:<15} {:<15}".format('KS-test (p-value)',
# 		                                         scifor(p_values[0],precision=p),
# 		                                         scifor(p_values[1],precision=p),
# 		                                         scifor(p_values[2],precision=p),
# 		                                         scifor(p_values[3],precision=p)))
# 	print("."*90)
# 	print("    Covariance matrices")
# 	print("."*90)
# 	print("Real samples:")
# 	print("")
# 	print('\n'.join([''.join(['{:<12.7f}'.format(item) for item in row]) 
# 	      for row in cov_real]))
# 	print("."*90)
# 	print("Fake samples:")
# 	print("")
# 	print('\n'.join([''.join(['{:<12.7f}'.format(item) for item in row]) 
# 	      for row in cov_fake]))
    
def print_results(m1, m2, s1, s2, p, sk1, sk2, ks, wd):

    print("_"*90)
    print("    Summary of results")
    print("_"*90)
    print("{:<20} {:<15} {:<15}".format('Parameter','Real samples','Fake samples'))
    print("_"*90)
    print("{:<20} {:<15.3f} {:<15.3f}".format('Mean', m1, m2))
    print("{:<20} {:<15.3f} {:<15.3f}".format('Std dev', s1, s2))
    print("{:<20} {:<15.3f} {:<15.3f}".format('Skewness', sk1, sk2))
    print("_"*90)
    print("{:<20} {:<15}".format('Pull', p))
    print("{:<20} {:<15}".format('KS-test (p-value)', ks))
    print("{:<20} {:<15}".format('Wasserstein distance', WASSERSTEIN_D))
	      
def evaluate(real, fake):
    m1, m2, s1, s2, p = get_mean_difference(real, fake)
    sk1, sk2 = get_skewness(real, fake)
    ks = ks_test(real, fake)
    wd = wasserstein_distance(real, fake)
    print_results(m1, m2, s1, s2, p, sk1, sk2, ks, wd)
