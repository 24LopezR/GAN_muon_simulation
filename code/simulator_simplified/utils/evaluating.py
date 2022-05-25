##################################################################################################
#### Evaluation module                                                                        ####
##################################################################################################
import numpy as np
from scipy.stats import skew, kstest
import matplotlib.pyplot as plt
from bitarray.util import int2ba

def decode(X):
    bitarrays = [list(int2ba(int(i), length=2)) for i in X]
    X_up = np.concatenate(bitarrays)
    return X_up

class Evaluation:
    
    def __init__(self, real_data, generated_data, n_values):
        self.real = real_data
        self.fake = generated_data
        self.n_values = n_values
        
    def calculate_parameters(self):
        
        self.means_real, self.means_fake, self.std_real, self.std_fake, self.pull = self.get_mean_difference()
        self.skew_real = skew(self.real)
        self.skew_fake = skew(self.fake)
        self.ks = kstest(self.real, self.fake)[1]
          	
    def get_mean_difference(self):
        means_real = np.mean(self.real, axis=0)
        means_fake = np.mean(self.fake, axis=0)
        err_real = np.std(self.real, axis=0, ddof=1)/np.sqrt(self.real.shape[0])
        err_fake = np.std(self.fake, axis=0, ddof=1)/np.sqrt(self.fake.shape[0])
        pull = (means_real - means_fake) / np.sqrt(err_real**2 + err_fake**2)
        return means_real, means_fake, np.std(self.real), np.std(self.fake), pull
        
    def print_results(self):
        print("_"*90)
        print("    Summary of results")
        print("_"*90)
        print("{:<20} {:<15} {:<15}".format('Parameter','Real samples','Fake samples'))
        print("_"*90)
        print("{:<20} {:<15.3f} {:<15.3f}".format('Mean', self.means_real, self.means_fake))
        print("{:<20} {:<15.3f} {:<15.3f}".format('Std dev', self.std_real, self.std_fake))
        print("{:<20} {:<15.3f} {:<15.3f}".format('Skewness', self.skew_real, self.skew_fake))
        print("_"*90)
        print("{:<20} {:<15}".format('Pull', self.pull))
        print("{:<20} {:<15}".format('KS-test (p-value)', self.ks))
        #print("{:<20} {:<15}".format('Wasserstein distance', WASSERSTEIN_D))
        
    def plot_hits(self):
        plt.rcParams["figure.figsize"] = (14,7)
        plt.rcParams["figure.titlesize"], plt.rcParams["axes.titlesize"] = (20,20)
        plt.rcParams["axes.labelsize"] = 18
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
        
        ax1.hist(self.real,  density = False, color = 'black', range=(1,self.n_values+1), 
                 bins=self.n_values, label='Real',      log=False)
        ax2.hist(self.fake,  density = False, color = 'red',   range=(1,self.n_values+1), 
                 bins=self.n_values, label='Generated', log=False)
        ax1.legend()
        ax2.legend()
        fig.suptitle('Number of hits');
        plt.show()
        return fig