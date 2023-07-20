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
from scipy.stats import skew, kstest

from plotting import plot_correlation_2Dhist


def load(inputfile):
    data = pd.read_csv(inputfile).to_numpy()
    # Select only some radius
    mask = [i in [4, 6, 8, 16, 18, 20] for i in data[:, 8]]

    data = data[mask]

    variables = data[:, :8]
    labels = data[:, 8]

    scaler = StandardScaler()
    weights = 1 / np.sqrt(data[:, 4] ** 2 + data[:, 5] ** 2)
    scaler.fit(variables, sample_weight=weights)
    variables = scaler.transform(variables)
    encoder = OneHotEncoder(sparse=False)
    encoder.fit(labels.reshape((-1, 1)))
    labels = encoder.transform(labels.reshape((-1, 1)))
    print(variables.shape)
    print(labels.shape)

    return np.concatenate([variables, labels], axis=1), scaler


class Evaluation:

    def __init__(self, generator_model, latent_dim, dataset, scaler):
        self.generator_model = generator_model
        self.latent_dim = latent_dim
        self.label_radius = dataset[:,8:]
        self.real_samples = dataset[:,4:8]
        self.input_data = dataset[:,0:4]
        self.scaler = scaler

    def generate_evaluation_samples(self):
        n_samples = self.input_data.shape[0]
        z_noise = np.random.normal(size=(n_samples, self.latent_dim))

        fake_samples = self.generator_model.predict([z_noise, np.hstack([self.input_data, self.label_radius])])

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
        print("{:<20} {:<15.3f} {:<15.3f} {:<15.3f} {:<15.3f}".format('KS-test (p-value)', p_values[0], p_values[1],
                                                                      p_values[2], p_values[3]))
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


def plot_difference(real, fake, x_range, scale, bins=200, title=''):
    plt.rcParams["figure.figsize"] = (14, 7)
    plt.rcParams["figure.titlesize"], plt.rcParams["axes.titlesize"] = (20, 20)
    plt.rcParams["axes.labelsize"] = 18

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    b = bins
    for real_samples in real:
        ax1.hist(real_samples, bins=b, density=False, range=x_range, histtype='step')
    for fake_samples in fake:
        ax2.hist(fake_samples, bins=b, density=False, range=x_range, histtype='step')
    ax1.set_xlabel(title+' real samples')
    ax2.set_xlabel(title+' generated samples')
    ax1.set_yscale(scale)
    ax2.set_yscale(scale)
    ax1.legend()
    ax2.legend()
    return fig


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

    LATENT_DIM = 64
    modelfile = '/home/ruben/WGAN-GP/wgangp_multi_radius/generator_model_200.h5'

    datafile = "/home/ruben/fewSamples/training_samples.csv"
    print('Loading model...')
    g_model = keras.models.load_model(modelfile)

    print('Loading_data...')
    dataset, scaler = load(datafile)
    masks = [(dataset[:, 8:].argmax(axis=1) == i) for i in range(0,6)]
    datasets = [dataset[m] for m in masks]

    eval_objects = [Evaluation(g_model, LATENT_DIM, dataset=data, scaler=scaler) for data in datasets]

    print('Generating evaluation samples...')
    for e in eval_objects:
        e.generate_evaluation_samples()

    print('Plotting results...')
    for e in eval_objects:
        e.evaluate(title="")

    real_datasets = [e.real_samples for e in eval_objects]
    fake_datasets = [e.fake_samples for e in eval_objects]

    output = 'alltest_200'
    out = PdfPages('/home/ruben/evaluation_' + output + '.pdf')
    out.savefig(plot_difference([r[:, 0] for r in real_datasets],
                                [f[:, 0] for f in fake_datasets],
                                (-25, 25), 'log', title='$\Delta x$'))
    out.savefig(plot_difference([r[:, 1] for r in real_datasets],
                                [f[:, 1] for f in fake_datasets],
                                (-25, 25), 'log', title='$\Delta y$'))
    out.savefig(plot_difference([r[:, 2] for r in real_datasets],
                                [f[:, 2] for f in fake_datasets],
                                (-1.5, 1.5), 'log', title='$\Delta {v_x}$'))
    out.savefig(plot_difference([r[:, 3] for r in real_datasets],
                                [f[:, 3] for f in fake_datasets],
                                (-1.5, 1.5), 'log', title='$\Delta {v_y}$'))

    out.savefig(plot_correlation_2Dhist(real_datasets[0][:, 0], real_datasets[0][:, 2],
                                        fake_datasets[0][:, 0], fake_datasets[0][:, 2],
                                        [[-15, 15], [-1, 1]], ['$\Delta x$ (4mm)', '$\Delta v_x$ (4mm)']))
    out.savefig(plot_correlation_2Dhist(real_datasets[0][:, 1], real_datasets[0][:, 3],
                                        fake_datasets[0][:, 1], fake_datasets[0][:, 3],
                                        [[-15, 15], [-1, 1]], ['$\Delta y$ (4mm)', '$\Delta v_y$ (4mm)']))
    out.savefig(plot_correlation_2Dhist(real_datasets[-1][:, 0], real_datasets[-1][:, 2],
                                        fake_datasets[-1][:, 0], fake_datasets[-1][:, 2],
                                        [[-15, 15], [-1, 1]], ['$\Delta x$ (20mm)', '$\Delta v_x$ (20mm)']))
    out.savefig(plot_correlation_2Dhist(real_datasets[-1][:, 1], real_datasets[-1][:, 3],
                                        fake_datasets[-1][:, 1], fake_datasets[-1][:, 3],
                                        [[-15, 15], [-1, 1]], ['$\Delta y$ (20mm)', '$\Delta v_y$ (20mm)']))
    out.close()
    print('Evaluation done!')