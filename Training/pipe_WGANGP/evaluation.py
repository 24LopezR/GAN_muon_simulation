import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend
import pandas as pd
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from optparse import OptionParser
from scipy.stats import skew, kstest

from plotting import plot_difference, plot_correlation_2Dhist


def load(inputfile):
    data = pd.read_csv(inputfile).to_numpy()
    variables = data[:,:8]
    labels = data[:,8]

    scaler = StandardScaler()
    weights = 1/np.sqrt(data[:,4]**2 + data[:,5]**2)
    scaler.fit(variables, sample_weight=weights)
    variables = scaler.transform(variables)

    return [variables, labels], scaler


class Evaluation:

    def __init__(self, generator_model, latent_dim, real_samples, input_data, scaler):
        self.generator_model = generator_model
        self.latent_dim = latent_dim
        self.real_samples = real_samples
        self.input_data = input_data
        self.scaler = scaler

    def generate_evaluation_samples(self):
        n_samples = self.input_data.shape[0]
        z_noise = np.random.normal(size=(n_samples, self.latent_dim))

        fake_samples = self.generator_model.predict([z_noise, self.input_data])

        self.fake_samples = self.scaler.inverse_transform(np.hstack([self.input_data, fake_samples]))[:, 4:]
        self.real_samples = self.scaler.inverse_transform(np.hstack([self.input_data, self.real_samples]))[:, 4:]

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
        p_values = [kstest(self.real_samples[:, 0], self.fake_samples[:, 0])[1],
                    kstest(self.real_samples[:, 1], self.fake_samples[:, 1])[1],
                    kstest(self.real_samples[:, 2], self.fake_samples[:, 2])[1],
                    kstest(self.real_samples[:, 3], self.fake_samples[:, 3])[1]]
        return p_values

    def evaluate(self):
        pull = self.get_mean_difference()
        cov = self.get_cov_matrices()
        cov_real, cov_fake, _ = cov
        skewness = self.get_skewness()
        skew_real, skew_fake, _ = skewness
        p_values = self.ks_test()

        print("."*100)
        print("    Summary of results")
        print("."*100)
        print("{:<20} {:<15} {:<15} {:<15} {:<15}".format('Parameter','Dx','Dy','Dv_x','Dv_y'))
        print("."*100)
        print("{:<20} {:<15.3f} {:<15.3f} {:<15.3f} {:<15.3f}".format('Pull', pull[0], pull[1], pull[2], pull[3]))
        print(" "*100)
        print("{:<20} {:<15.3f} {:<15.3f} {:<15.3f} {:<15.3f}".format('Skewness_real', skew_real[0], skew_real[1],
                                                                      skew_real[2], skew_real[3]))
        print("{:<20} {:<15.3f} {:<15.3f} {:<15.3f} {:<15.3f}".format('Skewness_fake', skew_fake[0], skew_fake[1],
                                                     skew_fake[2], skew_fake[3],))
        print(" "*100)
        print("{:<20} {:<15.3f} {:<15.3f} {:<15.3f} {:<15.3f}".format('KS-test (p-value)', p_values[0], p_values[1],
                                                     p_values[2], p_values[3]))
        print("."*100)
        print("    Covariance matrices")
        print("."*100)
        print("Real samples:")
        print("")
        print('\n'.join([''.join(['{:<12.7f}'.format(item) for item in row])
              for row in cov_real]))
        print("."*100)
        print("Fake samples:")
        print("")
        print('\n'.join([''.join(['{:<12.7f}'.format(item) for item in row])
              for row in cov_fake]))


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # GPU memory usage configuration
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    backend.set_session(sess)

    # Read user options
    parser = OptionParser(usage="%prog --help")
    parser.add_option("-m", "--model", dest="model", type="string", default=None,
                      help="Generator model to evaluate")
    parser.add_option("-l", "--latent", dest="latent_dim", type="int", default=200,
                      help="Latent dimension")
    (options, args) = parser.parse_args()

    LATENT_DIM = options.latent_dim
    modelfile = options.model

    datafile = "/home/ruben/fewSamples/training_samples.csv"
    print('Loading model...')
    g_model = keras.models.load_model(modelfile)

    print('Loading_data...')
    dataset, scaler = load(datafile)
    variables, labels = dataset
    variables = variables[labels==16]

    print('Generating evaluation samples...')
    e = Evaluation(g_model, LATENT_DIM, variables[:,4:8], variables[:,0:4], scaler)
    e.generate_evaluation_samples()

    print('Plotting results...')
    e.evaluate()
    real_dataset = e.real_samples
    fake_dataset = e.fake_samples
    output = 'test'
    out = PdfPages('evaluation_' + output + '.pdf')
    out.savefig(plot_difference(real_dataset[:,0],
                                fake_dataset[:,0],
                                (-40, 40), '$\Delta x$', 'log'))
    out.savefig(plot_difference(real_dataset[:,1],
                                fake_dataset[:,1],
                                (-40, 40), '$\Delta y$', 'log'))
    out.savefig(plot_difference(real_dataset[:,2],
                                fake_dataset[:,2],
                                (-1.5, 1.5), '$\Delta v_x$', 'log'))
    out.savefig(plot_difference(real_dataset[:,3],
                                fake_dataset[:,3],
                                (-1.5, 1.5), '$\Delta v_y$', 'log'))
    out.savefig(plot_correlation_2Dhist(real_dataset[:,0], real_dataset[:,2],
                                        fake_dataset[:,0], fake_dataset[:,2],
                                        [[-15, 15], [-1, 1]], ['$\Delta x$', '$\Delta v_x$']))
    out.savefig(plot_correlation_2Dhist(real_dataset[:,1], real_dataset[:,3],
                                        fake_dataset[:,1], fake_dataset[:,3],
                                        [[-15, 15], [-1, 1]], ['$\Delta y$', '$\Delta v_y$']))
    out.close()
    print('Evaluation done!')