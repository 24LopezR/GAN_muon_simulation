import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from matplotlib import pyplot
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from optparse import OptionParser
from scipy.stats import skew, kstest, wasserstein_distance, mannwhitneyu
import time

"""
This class implements all the functions to evaluate a certain generator model
"""
class Evaluation:

    def __init__(self, generator_model, latent_dim, dataset, scaler):
    """
    Arguments:
        - generator_model: path to .h5 file containing the model to evaluate
        - latent_dim     : dimension of latent space
        - dataset        : numpy array containing the evaluation data
                           format -> [ input_data, real_data_to_predict, label radius ]
        - scaler         : fitted StandardScaler object needed to normalize the evaluation dataset
    """
        self.generator_model = generator_model
        self.latent_dim = latent_dim
        self.input_data = dataset[:,0:4]
        self.real_samples = dataset[:,4:8]
        self.label_radius = dataset[:,8:]
        self.scaler = scaler
        self.inv_transform_real_samples()

    def inv_transform_real_samples(self):
        self.real_samples = self.scaler.inverse_transform(np.hstack([self.input_data, self.real_samples]))[:, 4:]

    def generate_evaluation_samples(self):
        # Delete samples
        self.fake_samples = None

        n_samples = self.input_data.shape[0]
        z_noise = np.random.normal(size=(n_samples, self.latent_dim))

        fake_samples = self.generator_model.predict([z_noise, np.hstack([self.input_data, self.label_radius])])

        self.fake_samples = self.scaler.inverse_transform(np.hstack([self.input_data, fake_samples]))[:, 4:]

    def get_mean_difference(self):
        means_real = np.mean(self.real_samples, axis=0)
        means_fake = np.mean(self.fake_samples, axis=0)
        err_real = np.std(self.real_samples, axis=0, ddof=1) / np.sqrt(self.real_samples.shape[0])
        err_fake = np.std(self.fake_samples, axis=0, ddof=1) / np.sqrt(self.fake_samples.shape[0])
        pull = (means_real - means_fake) / np.sqrt(err_real ** 2 + err_fake ** 2)
        return pull

    def get_cov_matrices(self):
        real_cov = np.cov(self.real_samples, rowvar=False)
        fake_cov = np.cov(self.fake_samples, rowvar=False)
        return real_cov, fake_cov, real_cov - fake_cov

    def get_skewness(self):
        skew_real = skew(self.real_samples)
        skew_fake = skew(self.fake_samples)
        return skew_real, skew_fake, skew_real - skew_fake

    def ks_test(self):
        p_values = []
        x_range = [(-40, 40), (-1, 1)]
        nreal, bins, _ = plt.hist(self.real_samples[:, 0][self.label_radius.argmax(1) == 0],
                               bins=100,
                               range=x_range[j // 2])
        print(nreal)
        return p_values

    def mannwhitneyu_test(self):
        stat, p_value = [], []
        for i in range(4):
            stat_temp, p_value_temp = mannwhitneyu(self.real_samples[:,i], self.fake_samples[:,i])
            stat.append(stat_temp)
            p_value.append(p_value_temp)
        return stat, p_value

    def evaluate(self, title):
        pull = self.get_mean_difference()
        cov = self.get_cov_matrices()
        cov_real, cov_fake, _ = cov
        skewness = self.get_skewness()
        skew_real, skew_fake, _ = skewness
        p_values = self.ks_test()

        print("." * 100)
        print("    Summary of results: "+title)
        print("." * 100)
        print("{:<20} {:<15} {:<15} {:<15} {:<15}".format('Parameter', 'Dx', 'Dy', 'Dv_x', 'Dv_y'))
        print("." * 100)
        print("{:<20} {:<15.3f} {:<15.3f} {:<15.3f} {:<15.3f}".format('Pull', pull[0], pull[1], pull[2], pull[3]))
        print(" " * 100)
        print("{:<20} {:<15.3f} {:<15.3f} {:<15.3f} {:<15.3f}".format('Skewness_real', skew_real[0], skew_real[1],
                                                                      skew_real[2], skew_real[3]))
        print("{:<20} {:<15.3f} {:<15.3f} {:<15.3f} {:<15.3f}".format('Skewness_fake', skew_fake[0], skew_fake[1],
                                                                      skew_fake[2], skew_fake[3], ))
        print(" " * 100)
        #print("{:<20} {:<15.3f} {:<15.3f} {:<15.3f} {:<15.3f}".format('KS-test (p-value)', p_values[0], p_values[1],
        #                                                              p_values[2], p_values[3]))
        print("." * 100)
        print("    Covariance matrices")
        print("." * 100)
        print("Real samples:")
        print("")
        print('\n'.join([''.join(['{:<12.7f}'.format(item) for item in row])
                         for row in cov_real]))
        print("." * 100)
        print("Fake samples:")
        print("")
        print('\n'.join([''.join(['{:<12.7f}'.format(item) for item in row])
                         for row in cov_fake]))
