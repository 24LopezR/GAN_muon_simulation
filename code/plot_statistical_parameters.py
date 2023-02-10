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
from tqdm import tqdm

def load_scaler():
    # -----------------------------------------------------------------------
    # This piece is needed to set up the scaler as in the training
    training_datafile = '/home/ruben/fewSamples/training_samples.csv'
    data = pd.read_csv(training_datafile).to_numpy()
    # Select only some radius
    mask = [i in [4, 6, 8, 10, 14, 16, 18, 20] for i in data[:, 8]]
    data = data[mask]

    variables = data[:, :8]

    scaler = StandardScaler()
    weights = 1 / np.sqrt(data[:, 4] ** 2 + data[:, 5] ** 2)
    scaler.fit(variables, sample_weight=weights)
    # -----------------------------------------------------------------------

    '''
    data = pd.read_csv(inputfile).to_numpy()
    mask = [i in radius for i in data[:, 8]]
    data = data[mask]
    variables = data[:, :8]
    labels = data[:, 8]
    variables = scaler.transform(variables)

    encoder = OneHotEncoder(sparse=False)
    encoder.fit(labels.reshape((-1, 1)))
    labels = encoder.transform(labels.reshape((-1, 1)))
    return np.concatenate([variables, labels], axis=1), scaler
    '''
    return scaler


def fill_results(eval, ntimes):
    pull_vec, p_value_vec = [], []
    cov_dxdvx, cov_dydvy = [], []
    print('    Start loop')
    for i in tqdm(range(ntimes)):
        eval.generate_evaluation_samples()
        #print(eval.real_samples.shape)
        #print(eval.fake_samples.shape)

        pull_temp = eval.get_mean_difference()
        pull_vec.append(pull_temp)

        _, _, cov_diff_temp = eval.get_cov_matrices()
        #print(cov_diff_temp[0,2])
        #print(cov_diff_temp[1,3])
        cov_dxdvx.append(cov_diff_temp[0,2])
        cov_dydvy.append(cov_diff_temp[1,3])

        stat, p_value_temp = eval.mannwhitneyu_test()
        p_value_vec.append(p_value_temp)
    return np.asarray(pull_vec), np.asarray(p_value_vec), np.asarray([cov_dxdvx, cov_dydvy]).T


def buildPullPlot(real_value, gen_values, title='', output=''):
    plt.rcParams["figure.figsize"] = (6, 6)
    plt.rcParams["figure.titlesize"], plt.rcParams["axes.titlesize"] = (20, 20)
    plt.rcParams["axes.labelsize"] = 18

    fig, ax1 = plt.subplots(1, 1)
    x_min = np.min([real_value] + gen_values.flatten())
    x_max = np.max([real_value] + gen_values.flatten())
    x_range = (x_min-1, x_max+1)
    bins = np.arange(x_min-1, x_max+1, 0.25)
    ax1.hist(gen_values,
               bins=bins,
               density=False,
               range=x_range,
               histtype='step',
               color=['dodgerblue', 'slateblue', 'violet', 'crimson'],
               linestyle='solid',
               label=['$\Delta x^*$',
                      '$\Delta y^*$',
                      '$\Delta v_x^*$',
                      '$\Delta v_y^*$'])
    ax1.axvline(x=real_value,
                color='black',
                linestyle='dashed',
                label='Real value')
    ax1.set_xlim(x_range)
    ax1.set_ylim([1.5*y for y in ax1.get_ylim()])
    ax1.legend()
    ax1.set_xlabel('Pull (adim.)', loc='right')
    ax1.set_ylabel('N times', loc='top')

    fig.suptitle(title)
    return fig


def buildCovPlot(real_value, cov_diffs, title='', output=''):
    plt.rcParams["figure.figsize"] = (6, 6)
    plt.rcParams["figure.titlesize"], plt.rcParams["axes.titlesize"] = (20, 20)
    plt.rcParams["axes.labelsize"] = 18

    fig, ax1 = plt.subplots(1, 1)
    x_min = np.min([0] + cov_diffs.flatten())
    x_max = np.max([0] + cov_diffs.flatten())
    x_range = (x_min-0.001, x_max+0.001)
    bins = np.arange(x_min-0.001, x_max+0.001, 0.0001)
    ax1.hist(cov_diffs,
               bins=bins,
               density=False,
               #range=x_range,
               histtype='step',
               color=['orangered', 'darkorange'],
               linestyle='solid',
               label=['$\Delta x^*$ - $\Delta v_x^*$',
                      '$\Delta y^*$ - $\Delta v_y^*$'])
    ax1.axvline(x=0,
                color='black',
                linestyle='dashed',
                label='Real value')
    ax1.set_xlim(x_range)
    ax1.set_ylim([1*y for y in ax1.get_ylim()])
    ax1.legend()
    ax1.set_xlabel('Covariance difference (mm$^2$)', loc='right')
    ax1.set_ylabel('N times', loc='top')

    fig.suptitle(title)
    return fig


def plot_test(conf_level, p_values, title='', output=''):
    plt.rcParams["figure.figsize"] = (6, 6)
    plt.rcParams["figure.titlesize"], plt.rcParams["axes.titlesize"] = (20, 20)
    plt.rcParams["axes.labelsize"] = 18

    fig, ax1 = plt.subplots(1, 1)
    x_min = 0 #np.min([conf_level] + p_values.flatten())
    x_max = np.max([conf_level] + p_values.flatten())
    x_range = (x_min, x_max+0.1)
    bins = np.arange(x_min, x_max+0.1, 0.01)
    ax1.hist(p_values,
               bins=bins,
               density=False,
               range=x_range,
               histtype='step',
               color=['dodgerblue', 'slateblue', 'violet', 'crimson'],
               linestyle='solid',
               label=['$\Delta x^*$',
                      '$\Delta y^*$',
                      '$\Delta v_x^*$',
                      '$\Delta v_y^*$'])
    ax1.axvline(x=conf_level,
                color='gray',
                linestyle='dashed',
                label='Confidence level')
    ax1.set_yscale('log')
    ax1.set_xlim(x_range)
    ax1.set_ylim([1*y for y in ax1.get_ylim()])
    ax1.legend()
    ax1.set_xlabel('p_value', loc='right')
    ax1.set_ylabel('N times', loc='top')

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
    parser.add_option("-m", "--mode", dest="mode", type="string", default='generate', help="Script run mode")
    (options, args) = parser.parse_args()

    ##############################################################################################################
    # Load generator model
    ##############################################################################################################
    LATENT_DIM = 16
    modelfile = '/home/ruben/final_models/WGANGP/generator_model_1000.h5'
    datafile = "/home/ruben/fewSamples_evaluation/evaluation_samples.csv"
    print('Loading model...')
    g_model = keras.models.load_model(modelfile, compile=True)

    ##############################################################################################################
    # Load scaler
    ##############################################################################################################
    radius = [4,6,8,10,14,16,18,20]
    scaler = load_scaler()

    ##############################################################################################################
    # Compute pull and plot results
    ##############################################################################################################
    N_TIMES = 500
    n = 300000
    WRITE_OUT = True
    write_opt = 'a'
    directory = '/home/ruben/GAN_muon_simulation/output/pval_test/'

    for i,rad in enumerate(radius):
        print('Processing r=' + str(rad) + ' mm')
        output = 'evaluation_stat_' + str(rad)

        if options.mode == 'generate':
            print('    Loading data...')
            dataset = pd.read_csv(datafile).to_numpy()
            data = dataset[dataset[:, 8] == rad][0:n, 0:8]
            data = scaler.transform(data)
            labels = np.zeros((n, 8))
            labels[:, i] = 1
            evals = Evaluation(g_model, LATENT_DIM,
                                            dataset=np.hstack([data, labels]),
                                            scaler=scaler)
            print('    Number of evaluation samples = ' + str(data.shape))

            print('    Generating samples...')
            pull_vector, p_value_vec, cov_diffs = fill_results(evals, ntimes=N_TIMES)

            if WRITE_OUT:
                print('    Writing .txt file...')
                # Save .txt
                with open(directory + output + '_pull.txt', write_opt) as f:
                    np.savetxt(f, pull_vector, delimiter=',')
                with open(directory + output + '_cov.txt', write_opt) as f:
                    np.savetxt(f, cov_diffs, delimiter=',')
                with open(directory + output + '_pval.txt', write_opt) as f:
                    np.savetxt(f, p_value_vec, delimiter=',')

        if options.mode == 'plot':
            print('    Reading .txt file...')
            pull_vector = pd.read_csv(directory + output + '_pull.txt', header=None).to_numpy()
            p_value_vec = pd.read_csv(directory + output + '_pval.txt', header=None).to_numpy()
            cov_diffs =   pd.read_csv(directory + output + '_cov.txt',  header=None).to_numpy()

            print('    Plotting results...')
            out_name = directory + output
            buildPullPlot(real_value=0,
                          gen_values=pull_vector,
                          title='Pull distributions ($r='+str(rad)+'$ mm)').savefig(out_name + '_pull.png')
            buildCovPlot(real_value=0,
                         cov_diffs=cov_diffs,
                         title='Covariance differences ($r=' + str(rad) + '$ mm)').savefig(out_name + '_cov.png')
            plot_test(conf_level=0.05,
                      p_values=p_value_vec,
                      title='p_value distribution ($r='+str(rad)+'$ mm)').savefig(out_name + '_test.png')

    # Eval interpolation
    print('Processing r=12 mm')
    output = 'evaluation_stat_12_interpolation'

    if options.mode == 'generate':
        print('    Loading data...')
        # Interpolation samples
        interpolation_data = pd.read_csv(datafile).to_numpy()
        interpolation_data = interpolation_data[interpolation_data[:, 8] == 12][0:n, 0:8]
        interpolation_data = scaler.transform(interpolation_data)
        interpolation_labels = np.zeros((n, 8))
        interpolation_labels[:, 3:5] = 0.5
        eval_interpolation = Evaluation(g_model, LATENT_DIM,
                                        dataset=np.hstack([interpolation_data, interpolation_labels]),
                                        scaler=scaler)
        print('    Number of evaluation samples = ' + str(interpolation_data.shape))

        print('    Generating samples...')
        pull_vector, p_value_vec, cov_diffs = fill_results(evals, ntimes=N_TIMES)

        if WRITE_OUT:
            print('    Writing .txt file...')
            # Save .txt
            with open(directory + output + '_pull.txt', write_opt) as f:
                np.savetxt(f, pull_vector, delimiter=',')
            with open(directory + output + '_cov.txt', write_opt) as f:
                np.savetxt(f, cov_diffs, delimiter=',')
            with open(directory + output + '_pval.txt', write_opt) as f:
                np.savetxt(f, p_value_vec, delimiter=',')

    if options.mode == 'plot':
        print('    Reading .txt file...')
        pull_vector = pd.read_csv(directory + output + '_pull.txt', header=None).to_numpy()
        p_value_vec = pd.read_csv(directory + output + '_pval.txt', header=None).to_numpy()
        cov_diffs =   pd.read_csv(directory + output + '_cov.txt',  header=None).to_numpy()

        print('    Plotting results...')
        out_name = directory + output
        buildPullPlot(real_value=0,
                      gen_values=pull_vector,
                      title='Pull distributions ($r=12$ mm)').savefig(out_name+'_pull.png')
        buildCovPlot(real_value=0,
                      cov_diffs=cov_diffs,
                      title='Covariance differences ($r=12$ mm)').savefig(out_name+'_cov.png')
        plot_test(conf_level=0.05,
                      p_values=p_value_vec,
                      title='p_value distribution ($r=12$ mm)').savefig(out_name+'_test.png')

    print('Evaluation done!')

