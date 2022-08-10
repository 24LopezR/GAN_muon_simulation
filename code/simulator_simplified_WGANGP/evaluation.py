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
from scipy.stats import wasserstein_distance

from plotting import plot_difference, plot_correlation_2Dhist
from layers.gumbel_softmax import GumbelSoftmaxActivation
from layers.adapstep import AdaptativeStepActivation

class Evaluation:

    def __init__(self, generator_model, latent_dim):
        self.generator_model = generator_model
        self.latent_dim = latent_dim
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder()


    def load(self, inputfile):
        data = pd.read_csv(inputfile).to_numpy()
        in_data = data[:, 0:2]
        first_active = data[:, 2].astype(int)
        last_active = data[:, 3].astype(int)
        self.real_samples = np.zeros((data.shape[0], 216))
        for i in range(data.shape[0]):
            self.real_samples[i, first_active[i]:(last_active[i] + 1)] = 1

        # scale input data (x, vx)
        weights = 1 / np.sqrt(in_data[:, 0] ** 2 + in_data[:, 1] ** 2)
        self.scaler.fit(in_data, sample_weight=weights)
        self.in_data_scaled = self.scaler.transform(in_data)

        #self.encoder.fit(self.real_samples.reshape(-1,1))


    def generate_evaluation_samples(self):
        n_samples = self.in_data_scaled.shape[0]
        z_noise = np.random.normal(size=(n_samples, self.latent_dim))

        fake_samples = self.generator_model.predict([z_noise, self.in_data_scaled])
        self.fake_samples_round = np.around(fake_samples / fake_samples.max(axis=1, keepdims=True))
        #self.fake_samples_round = np.around(fake_samples)
        self.fake_samples = fake_samples

def plot_hist(real, fake, b, title=''):
    real_sum = real.sum(axis=1)
    fake_sum = fake.sum(axis=1)

    plt.rcParams["figure.figsize"] = (14, 7)
    plt.rcParams["figure.titlesize"], plt.rcParams["axes.titlesize"] = (20, 20)
    plt.rcParams["axes.labelsize"] = 18
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)

    nsreal, binsr, _ = ax1.hist(real_sum, bins=b, range=(0,b), color='black', label='Data')
    nsfake, binsf, _ = ax2.hist(fake_sum, bins=b, range=(0,b), color='green', label='Generated')
    ax1.legend()
    ax2.legend()
    WD = wasserstein_distance(nsreal, nsfake)
    fig.suptitle(title+' WD = {:<10.3f}'.format(WD))
    return fig


if __name__ == "__main__":

    os. environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
    datafile = "/home/ruben/GAN_muon_simulation/data/sim_start_final.csv"

    print('Loading model...')
    g_model = keras.models.load_model(modelfile,
                                      custom_objects = {'AdaptativeStepActivation':
                                                            AdaptativeStepActivation()})

    e = Evaluation(g_model, LATENT_DIM)
    print('Loading_data...')
    e.load(datafile)
    print('Generating evaluation samples...')
    e.generate_evaluation_samples()

    print('Plotting results...')
    real_dataset = e.real_samples
    fake_dataset = e.fake_samples
    fake_dataset_round = e.fake_samples_round
    for i in range(fake_dataset_round.shape[0]):
        if np.sum(fake_dataset_round[i])==0:
            print(real_dataset[i])
            print(fake_dataset[i])
            break
    output = 'test'
    out = PdfPages('evaluation_' + output + '.pdf')
    out.savefig(plot_hist(real_dataset, fake_dataset_round, 15))
    #out.savefig(plot_hist(real_dataset[:,1], fake_dataset[:,1], 1070, title='Last wire'))
    out.close()

    # Mean distance between activated wires

    print('Evaluation done!')