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
from scipy.stats import skew, kstest, wasserstein_distance
import time


def load(inputfile):
    training_datafile = '/home/ruben/fewSamples/training_samples.csv'
    data = pd.read_csv(training_datafile).to_numpy()
    # Select only some radius
    mask = [i in [4, 6, 8, 10, 14, 16, 18, 20] for i in data[:, 8]]
    #mask = [i in [4, 6, 8, 16, 18, 20] for i in data[:, 8]]
    data = data[mask]

    variables = data[:, :8]

    scaler = StandardScaler()
    weights = 1 / np.sqrt(data[:, 4] ** 2 + data[:, 5] ** 2)
    scaler.fit(variables, sample_weight=weights)

    data = pd.read_csv(inputfile).to_numpy()
    mask = [i in [4, 6, 8, 10, 14, 16, 18, 20] for i in data[:, 8]]
    data = data[mask]
    variables = data[:, :8]
    labels = data[:, 8]
    variables = scaler.transform(variables)

    encoder = OneHotEncoder(sparse=False)
    encoder.fit(labels.reshape((-1, 1)))
    labels = encoder.transform(labels.reshape((-1, 1)))
    return np.concatenate([variables, labels], axis=1), scaler


class Evaluation:

    def __init__(self, generator_model, latent_dim, dataset, scaler):
        self.generator_model = generator_model
        self.latent_dim = latent_dim
        self.input_data = dataset[:,0:4]
        self.real_samples = dataset[:,4:8]
        self.label_radius = dataset[:,8:]
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


def plot_difference(real, fake, nbins=200, title=''):
    plt.rcParams["figure.figsize"] = (28, 9)
    plt.rcParams["figure.titlesize"], plt.rcParams["axes.titlesize"] = (20, 20)
    plt.rcParams["axes.labelsize"] = 18

    fig, ((ax1, ax2, ax3, ax4), (ax1r, ax2r, ax3r, ax4r)) = plt.subplots(2, 4, sharex=False,
                                                                         gridspec_kw={'height_ratios': [3, 1]})
    axes = [ax1, ax2, ax3, ax4]
    axesr = [ax1r, ax2r, ax3r, ax4r]
    x_range = [(-40,40), (-1,1)]
    labels = ['$\Delta x$', '$\Delta y$', '$\Delta v_x$', '$\Delta v_y$']
    for i in range(4):
        ns, bins, patches = axes[i].hist([real[:,i], fake[:,i]],
                                         bins=nbins,
                                         density=False,
                                         range=x_range[i//2],
                                         histtype='step',
                                         color=['black','red'],
                                         label=['data','generated'])
        axes[i].set_yscale('log')
        axes[i].set_xlim(left=x_range[i // 2][0], right=x_range[i // 2][1])
        axes[i].legend()
        WD = wasserstein_distance(real[:,i], fake[:,i])

        ## Ratio plot
        r = []
        b = []
        r_xerror = []
        r_yerror = []
        y1 = ns[0]
        y2 = ns[1]
        for n in range(0, len(y1)):
            if y1[n] == 0 or y2[n] == 0:
                r.append(np.nan)
                r_xerror.append(np.nan)
                r_yerror.append(np.nan)
                b.append(bins[n] + (bins[n + 1] - bins[n]) / 2.)
                continue
            r.append(y1[n] / y2[n])
            r_xerror.append((bins[n + 1] - bins[n]) / 2.)  # La anchura del bin
            r_yerror.append(((y1[n] / y1[n] ** 2) + (y2[n] / y2[
                n] ** 2)) ** 0.5)  # Suma en cuadratura de errores (error en un histograma es la raiz del n'umero de cuentas)
            b.append(bins[n] + (bins[n + 1] - bins[n]) / 2.)

        axesr[i].errorbar(x = b, y = r, yerr=r_yerror, xerr=r_xerror, fmt = 'o', color = 'k', ecolor = 'k')
        axesr[i].set_xlim(left=x_range[i//2][0], right=x_range[i//2][1])
        axesr[i].set_ylim(bottom=0, top=2)
        axesr[i].set_xlabel(labels[i] + ' (WD = {:<10.5f})'.format(WD))

    fig.suptitle(title)
    return fig

def plot_interpolation(real, fake, limit_up, limit_down, nbins=200, title=''):
    plt.rcParams["figure.figsize"] = (28, 9)
    plt.rcParams["figure.titlesize"], plt.rcParams["axes.titlesize"] = (20, 20)
    plt.rcParams["axes.labelsize"] = 18

    fig, ((ax1, ax2, ax3, ax4), (ax1r, ax2r, ax3r, ax4r)) = plt.subplots(2, 4, sharex=False,
                                                                         gridspec_kw={'height_ratios': [3, 1]})
    axes = [ax1, ax2, ax3, ax4]
    axesr = [ax1r, ax2r, ax3r, ax4r]
    x_range = [(-40,40), (-1,1)]
    labels = ['$\Delta x$', '$\Delta y$', '$\Delta v_x$', '$\Delta v_y$']
    for i in range(4):
        axes[i].hist([limit_up[:, i], limit_down[:, i]],
                     bins=nbins,
                     density=False,
                     range=x_range[i // 2],
                     histtype='step',
                     color=['pink', 'pink'],
                     label=['generated (r=14mm)', 'generated (r=10mm)'])
        ns, bins, patches = axes[i].hist([real[:,i],fake[:, i]],
                                         bins=nbins,
                                         density=False,
                                         range=x_range[i//2],
                                         histtype='step',
                                         color=['black','red'],
                                         label=['data','generated'])
        axes[i].set_yscale('log')
        axes[i].set_xlim(left=x_range[i // 2][0], right=x_range[i // 2][1])
        axes[i].legend()
        WD = wasserstein_distance(real[:,i], fake[:,i])

        ## Ratio plot
        r = []
        b = []
        r_xerror = []
        r_yerror = []
        y1 = ns[0]
        y2 = ns[1]
        for n in range(0, len(y1)):
            if y1[n] == 0 or y2[n] == 0:
                r.append(np.nan)
                r_xerror.append(np.nan)
                r_yerror.append(np.nan)
                b.append(bins[n] + (bins[n + 1] - bins[n]) / 2.)
                continue
            r.append(y1[n] / y2[n])
            r_xerror.append((bins[n + 1] - bins[n]) / 2.)  # La anchura del bin
            r_yerror.append(((y1[n] / y1[n] ** 2) + (y2[n] / y2[
                n] ** 2)) ** 0.5)  # Suma en cuadratura de errores (error en un histograma es la raiz del n'umero de cuentas)
            b.append(bins[n] + (bins[n + 1] - bins[n]) / 2.)

        axesr[i].errorbar(x=b, y=r, yerr=r_yerror, xerr=r_xerror, fmt='o', color='k', ecolor='k')
        axesr[i].set_xlim(left=x_range[i // 2][0], right=x_range[i // 2][1])
        axesr[i].set_ylim(bottom=0, top=2)
        axesr[i].set_xlabel(labels[i] + ' (WD = {:<10.5f})'.format(WD))

    fig.suptitle(title)
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

    LATENT_DIM = options.latent_dim
    modelfile = options.model

    datafile = "/home/ruben/fewSamples_evaluation/evaluation_samples.csv"
    print('Loading model...')
    g_model = keras.models.load_model(modelfile)

    print('Loading_data...')
    dataset, scaler = load(datafile)
    n = 300000
    data_reduced = np.zeros((8*n,dataset.shape[1]))
    for i in range(8):
        data_reduced[n*i:n*(i+1)] = dataset[dataset[:,8:].argmax(1)==i][0:n]
    dataset = data_reduced
    print('Number of evaluation samples = '+str(dataset.shape[0]))

    interpolation_data = pd.read_csv(datafile).to_numpy()
    interpolation_data = interpolation_data[interpolation_data[:,8]==12][0:n,0:8]
    interpolation_data = scaler.transform(interpolation_data)
    interpolation_labels = np.zeros((n,8))
    interpolation_labels[:,3:5] = 0.5

    eval_interpolation = Evaluation(g_model, LATENT_DIM, dataset=np.hstack([interpolation_data, interpolation_labels]),
                                    scaler=scaler)
    eval = Evaluation(g_model, LATENT_DIM, dataset=dataset, scaler=scaler)

    print('Generating evaluation samples...')
    start_time = time.time()
    print('    Start time = '+str(start_time))
    eval_interpolation.generate_evaluation_samples()
    eval.generate_evaluation_samples()
    print('    End time = '+str(time.time()))
    print("--- Generation time: %s seconds ---" % (time.time() - start_time))
    real_dataset = eval.real_samples
    fake_dataset = eval.fake_samples
    labels = eval.label_radius
    real_inter = eval_interpolation.real_samples
    fake_inter = eval_interpolation.fake_samples

    print('Plotting results...')
    output = 'test_ratio'
    out = PdfPages('evaluation_' + output + '.pdf')
    num_classes = labels.shape[1]
    radius = [4,6,8,10,14,16,18,20]
    for j in range(num_classes):
        out.savefig(plot_difference(real_dataset[labels.argmax(1)==j],
                                    fake_dataset[labels.argmax(1)==j],
                                    nbins=100,
                                    title='r = '+str(radius[j])+'mm'))
    out.savefig(plot_interpolation(real_inter,
                                   fake_inter,
                                   limit_down=fake_dataset[labels.argmax(1)==3],
                                   limit_up=fake_dataset[labels.argmax(1)==4],
                                   nbins=100,
                                   title='r = 12mm'))

    #out.savefig(plot_correlation_2Dhist(real_dataset_20[:, 0], real_dataset_20[:, 2],
    #                                    fake_dataset_20[:, 0], fake_dataset_20[:, 2],
    #                                    [[-15, 15], [-1, 1]], ['$\Delta x$ (6mm)', '$\Delta v_x$ (6mm)']))
    #out.savefig(plot_correlation_2Dhist(real_dataset_04[:, 1], real_dataset_04[:, 3],
    #                                    fake_dataset_04[:, 1], fake_dataset_04[:, 3],
    #                                    [[-15, 15], [-1, 1]], ['$\Delta y$ (18mm)', '$\Delta v_y$ (18mm)']))
    out.close()
    print('Evaluation done!')
