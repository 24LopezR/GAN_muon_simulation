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
from Evaluation import Evaluation


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


def plot_difference(real, fake, nbins=200, title=''):
    plt.rcParams["figure.figsize"] = (28, 10)
    plt.rcParams["figure.titlesize"], plt.rcParams["axes.titlesize"] = (20, 20)
    plt.rcParams["axes.labelsize"] = 18

    fig, ((ax1, ax2, ax3, ax4), (ax1r, ax2r, ax3r, ax4r)) = plt.subplots(2, 4, sharex=False,
                                                                         gridspec_kw={'height_ratios': [3, 1]})
    axes = [ax1, ax2, ax3, ax4]
    axesr = [ax1r, ax2r, ax3r, ax4r]
    x_range = [(-40,40), (-1,1)]
    labels = ['$\Delta x$', '$\Delta y$', '$\Delta v_x$', '$\Delta v_y$']
    for i in range(4):
        ns_1, bins_1, _ = axes[i].hist([real[:300000,i], real[-300000:,i]],
                                        bins=nbins,
                                        density=False,
                                        range=x_range[i//2],
                                        histtype='step',
                                        color=['blue','red'],
                                        linestyle = 'solid',
                                        label=['data 4 mm','data 20 mm'])
        ns_2, bins_2, _ = axes[i].hist([fake[:300000, i], fake[-300000:, i]],
                     bins=nbins,
                     density=False,
                     range=x_range[i//2],
                     histtype='step',
                     color=['blue','red'],
                     linestyle='dashed',
                     label=['generated 4 mm', 'generated 20 mm'])
        axes[i].set_yscale('log')
        axes[i].set_xlim(left=x_range[i // 2][0], right=x_range[i // 2][1])
        axes[i].legend()
        axes[i].set_xlabel(labels[i])

        ## Ratio plot 1
        r = []
        b = []
        r_xerror = []
        r_yerror = []
        y1 = ns_1[0]
        y2 = ns_2[0]
        for n in range(0, len(y1)):
            if y1[n] == 0 or y2[n] == 0:
                r.append(np.nan)
                r_xerror.append(np.nan)
                r_yerror.append(np.nan)
                b.append(bins_1[n] + (bins_1[n + 1] - bins_1[n]) / 2.)
                continue
            r.append(y1[n] / y2[n])
            r_xerror.append((bins_1[n + 1] - bins_1[n]) / 2.)  # La anchura del bin
            r_yerror.append(((y1[n] / y1[n] ** 2) + (y2[n] / y2[
                n] ** 2)) ** 0.5)  # Suma en cuadratura de errores (error en un histograma es la raiz del n'umero de cuentas)
            b.append(bins_1[n] + (bins_1[n + 1] - bins_1[n]) / 2.)

        axesr[i].errorbar(x=b, y=r, yerr=r_yerror, xerr=r_xerror, fmt='o', color='blue', ecolor='blue')

        ## Ratio plot 2
        r = []
        b = []
        r_xerror = []
        r_yerror = []
        y1 = ns_1[1]
        y2 = ns_2[1]
        for n in range(0, len(y1)):
            if y1[n] == 0 or y2[n] == 0:
                r.append(np.nan)
                r_xerror.append(np.nan)
                r_yerror.append(np.nan)
                b.append(bins_2[n] + (bins_2[n + 1] - bins_2[n]) / 2.)
                continue
            r.append(y1[n] / y2[n])
            r_xerror.append((bins_2[n + 1] - bins_2[n]) / 2.)  # La anchura del bin
            r_yerror.append(((y1[n] / y1[n] ** 2) + (y2[n] / y2[
                n] ** 2)) ** 0.5)  # Suma en cuadratura de errores (error en un histograma es la raiz del n'umero de cuentas)
            b.append(bins_2[n] + (bins_2[n + 1] - bins_2[n]) / 2.)
        axesr[i].axhline(y=1, linestyle='dashed', color='black')
        axesr[i].errorbar(x=b, y=r, yerr=r_yerror, xerr=r_xerror, fmt='o', color='red', ecolor='red')
        axesr[i].set_xlim(left=x_range[i // 2][0], right=x_range[i // 2][1])
        axesr[i].set_ylim(bottom=0, top=2)
        axesr[i].set_xlabel(labels[i])

    fig.suptitle(title)
    return fig


def plot_interpolation(real, fake, limit_up, limit_down, nbins=200, title=''):
    plt.rcParams["figure.figsize"] = (28, 10)
    plt.rcParams["figure.titlesize"], plt.rcParams["axes.titlesize"] = (20, 20)
    plt.rcParams["axes.labelsize"] = 18

    fig, ((ax1, ax2, ax3, ax4), (ax1r, ax2r, ax3r, ax4r)) = plt.subplots(2, 4, sharex=False, gridspec_kw={'height_ratios': [3, 1]})
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
        axesr[i].axhline(y=1, linestyle='dashed', color='black')
        axesr[i].errorbar(x=b, y=r, yerr=r_yerror, xerr=r_xerror, fmt='o', color='k', ecolor='k')
        axesr[i].set_xlim(left=x_range[i // 2][0], right=x_range[i // 2][1])
        axesr[i].set_ylim(bottom=0, top=2)
        axesr[i].set_xlabel(labels[i])
    return fig


def compute_WD(real, fake):
    WD_matrix = np.zeros([4,8,8])
    for k in range(4):
        for i in range(8):
            for j in range(8):
                if i == j:
                    WD_matrix[k, i, j] = wasserstein_distance(real[300000 * i:300000 * (i + 1), k],
                                                              fake[300000 * j:300000 * (j + 1), k])
                else:
                    WD_matrix[k, i, j] = wasserstein_distance(real[300000*i:300000*(i+1),k],
                                                              real[300000*j:300000*(j+1),k])

    return WD_matrix


def plot_WD(WD_matrix):
    plt.rcParams["figure.figsize"] = (28, 7)
    plt.rcParams["figure.titlesize"], plt.rcParams["axes.titlesize"] = (20, 20)
    plt.rcParams["axes.labelsize"] = 18
    plt.xlabel('xlabel', fontsize=18)
    plt.ylabel('ylabel', fontsize=18)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharex=False, gridspec_kw={'width_ratios': [1,1,1,1]})
    fig.suptitle('Wasserstein distance between distributions', fontsize=20)
    axes = [ax1, ax2, ax3, ax4]
    labels = ['$\Delta x$', '$\Delta y$', '$\Delta v_x$', '$\Delta v_y$']
    for k in range(4):
        a = axes[k].matshow(WD_matrix[k]/np.max(WD_matrix[k]), cmap=plt.cm.Blues)
        for i in range(8):
            for j in range(8):
                if i == j:
                    c = np.around(WD_matrix[k, i, j]/np.max(WD_matrix[k]), 3)
                    axes[k].text(i, j, str(c), va='center', ha='center')
        axes[k].set_xlabel(labels[k])
        axes[k].set_xticklabels([2,4,6,8,10,14,16,18,20])
        axes[k].set_yticklabels([2,4,6,8,10,14,16,18,20])
        axes[k].tick_params(axis='both', which='major', labelsize=12)
    fig.colorbar(a, ax=axes, location = 'bottom', fraction = 0.05)

    return fig


def compute_covariances(real, fake):
    cov_dxdvx = np.zeros([8,9])
    cov_dydvy = np.zeros([8,9])
    for i in range(8):
        for j in range(8):
            cov_dxdvx[i, -1] = np.cov([fake[300000 * i:300000 * (i + 1), 0],
                                      fake[300000 * i:300000 * (i + 1), 2]])[0,1]
            cov_dydvy[i, -1] = np.cov([fake[300000 * i:300000 * (i + 1), 0],
                                      fake[300000 * i:300000 * (i + 1), 2]])[0,1]

            cov_dxdvx[i, j] = np.cov([real[300000 * i:300000 * (i + 1), 0],
                                      real[300000 * j:300000 * (j + 1), 2]])[0,1]
            cov_dydvy[i, j] = np.cov([real[300000 * i:300000 * (i + 1), 0],
                                      real[300000 * j:300000 * (j + 1), 2]])[0,1]

    return cov_dxdvx, cov_dydvy


def plot_covs(cov_dxdvx, cov_dydvy):
    plt.rcParams["figure.figsize"] = (14, 7)
    plt.rcParams["figure.titlesize"], plt.rcParams["axes.titlesize"] = (20, 20)
    plt.rcParams["axes.labelsize"] = 18

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=False, gridspec_kw={'width_ratios': [1,1]})
    fig.suptitle('Correlations between distributions', fontsize=20)
    axes = [ax1, ax2]
    labels = ['$|\Delta x - \Delta v_x|$ (normalized)', '$|\Delta y - \Delta v_y|$ (normalized)']
    cov_dxdvx = abs(cov_dxdvx) / np.max(abs(cov_dxdvx))
    cov_dydvy = abs(cov_dydvy) / np.max(abs(cov_dydvy))
    ax1.matshow(cov_dxdvx[:,:8], cmap=plt.cm.Greens)
    a = ax2.matshow(cov_dydvy[:,:8], cmap=plt.cm.Greens)
    for k in range(2):
        axes[k].set_xlabel(labels[k])
        axes[k].set_xticklabels([2,4,6,8,10,14,16,18,20])
        axes[k].set_yticklabels([2,4,6,8,10,14,16,18,20])
    fig.colorbar(a, ax=axes, location = 'bottom', fraction = 0.05)

    return fig


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # GPU memory usage configuration
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    backend.set_session(sess)

    # Read user options
    LATENT_DIM = 16
    modelfile = '/home/ruben/final_models/WGANGP/generator_model_1000.h5'

    datafile = "/home/ruben/fewSamples_evaluation/evaluation_samples.csv"
    print('Loading model...')
    g_model = keras.models.load_model(modelfile)

    LOAD=1
    if LOAD:
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
        eval_ = Evaluation(g_model, LATENT_DIM, dataset=dataset, scaler=scaler)

        print('Generating evaluation samples...')
        start_time = time.time()
        print('    Start time = '+str(start_time))
        eval_interpolation.generate_evaluation_samples()
        eval_.generate_evaluation_samples()
        print('    End time = '+str(time.time()))
        print("--- Generation time: %s seconds ---" % (time.time() - start_time))
        real_dataset = eval_.real_samples
        fake_dataset = eval_.fake_samples
        labels = eval_.label_radius
        real_inter = eval_interpolation.real_samples
        fake_inter = eval_interpolation.fake_samples

    # Calculate means
    means_real = np.mean(real_inter, axis=0)
    means_fake = np.mean(fake_inter, axis=0)

    # Calculate skewness
    skew_real = skew(real_inter)
    skew_fake = skew(fake_inter)

    # Calculate covariance matrices
    real_cov = np.cov(real_inter, rowvar=False)
    fake_cov = np.cov(fake_inter, rowvar=False)

    print("." * 90)
    print("    Summary of results: Interpolation 12 mm")
    print("." * 90)
    print("{:<20} {:<15} {:<15} {:<15} {:<15}".format('Parameter', 'Dx', 'Dy', 'Dv_x', 'Dv_y'))
    print("." * 90)
    print("{:<20} {:<15.7e} {:<15.7e} {:<15.7e} {:<15.7e}".format('Mean real', means_real[0], means_real[1],
                                                                  means_real[2], means_real[3]))
    print(
        "{:<20} {:<15.7e} {:<15.7e} {:<15.7e} {:<15.7e}".format('Mean gen', means_fake[0], means_fake[1], means_fake[2],
                                                                means_fake[3]))
    print("{:<20} {:<15.7f} {:<15.7f} {:<15.7f} {:<15.7f}".format('Skew real', skew_real[0], skew_real[1], skew_real[2],
                                                                  skew_real[3]))
    print("{:<20} {:<15.7f} {:<15.7f} {:<15.7f} {:<15.7f}".format('Skew gen', skew_fake[0], skew_fake[1], skew_fake[2],
                                                                  skew_fake[3]))
    print("." * 90)
    print("    Covariance matrices")
    print("." * 90)
    print("Real samples:")
    print("")
    print('\n'.join([''.join(['{:<12.7f}'.format(item) for item in row])
                     for row in real_cov]))
    print("." * 90)
    print("Fake samples:")
    print("")
    print('\n'.join([''.join(['{:<12.7f}'.format(item) for item in row])
                     for row in fake_cov]))

    if LOAD: WD_matrix = compute_WD(real_dataset, fake_dataset)
    if LOAD: cov_dxdvx, cov_dydvy = compute_covariances(real_dataset, fake_dataset)

    print('Plotting results...')
    output = 'paper'
    out = PdfPages('/home/ruben/GAN_muon_simulation/output/evaluation_' + output + '.pdf')
    num_classes = labels.shape[1]
    out.savefig(plot_difference(real_dataset,
                               fake_dataset,
                               nbins=100,
                               title=''))
    out.savefig(plot_interpolation(real_inter,
                                  fake_inter,
                                  limit_down=fake_dataset[labels.argmax(1) == 3],
                                  limit_up=fake_dataset[labels.argmax(1) == 4],
                                  nbins=100,
                                  title='r = 12mm'))
    out.savefig(plot_WD(WD_matrix))
    out.savefig(plot_covs(cov_dxdvx, cov_dydvy))
    out.close()
    print('Evaluation done!')
