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
from joblib import load
from Common.Constants import WEIGHTED_SCALER, EVALUATION_SAMPLES_PATH, MODEL_PATH
from Cucconi import cucconi_test

############################################################################
### Constants
OUTPUT_FILE = './evaluationPaper.pdf'
############################################################################

def plot_difference(real, fake, nbins=200, title=''):
    plt.rcParams["figure.figsize"] = (7, 10)
    plt.rcParams["figure.titlesize"], plt.rcParams["axes.titlesize"] = (20, 20)
    plt.rcParams["axes.labelsize"] = 18

    name = ['deltax', 'deltay', 'deltavx', 'deltavy']
    x_range = [(-40,40), (-1,1)]
    labels = ['$\Delta x$', '$\Delta y$', '$\Delta v_x$', '$\Delta v_y$']
    for i in range(4):
        fig, ((ax1), (ax1r)) = plt.subplots(2, 1, sharex=False,
                                            gridspec_kw={'height_ratios': [3, 1]})
        
        ns_1, bins_1, _ = ax1.hist([real[:300000,i], real[-300000:,i]],
                                        bins=nbins,
                                        density=False,
                                        range=x_range[i//2],
                                        histtype='step',
                                        color=['blue','red'],
                                        linestyle = 'solid',
                                        label=['data 4 mm','data 20 mm'])
        ns_2, bins_2, _ = ax1.hist([fake[:300000, i], fake[-300000:, i]],
                     bins=nbins,
                     density=False,
                     range=x_range[i//2],
                     histtype='step',
                     color=['blue','red'],
                     linestyle='dashed',
                     label=['generated 4 mm', 'generated 20 mm'])
        ax1.set_yscale('log')
        ax1.set_xlim(left=x_range[i // 2][0], right=x_range[i // 2][1])
        ax1.legend()
        ax1.set_xlabel(labels[i])

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

        ax1r.errorbar(x=b, y=r, yerr=r_yerror, xerr=r_xerror, fmt='o', color='blue', ecolor='blue')

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
        ax1r.axhline(y=1, linestyle='dashed', color='black')
        ax1r.errorbar(x=b, y=r, yerr=r_yerror, xerr=r_xerror, fmt='o', color='red', ecolor='red')
        ax1r.set_xlim(left=x_range[i // 2][0], right=x_range[i // 2][1])
        ax1r.set_ylim(bottom=0, top=2)
        ax1r.set_xlabel(labels[i])
        plt.savefig(f'diff_ratio_{name[i]}.png', format='png')
        plt.savefig(f'diff_ratio_{name[i]}.pdf', format='pdf')


def plot_interpolation(real, fake, limit_up, limit_down, nbins=200, title=''):
    plt.rcParams["figure.figsize"] = (7, 10)
    plt.rcParams["figure.titlesize"], plt.rcParams["axes.titlesize"] = (20, 20)
    plt.rcParams["axes.labelsize"] = 18

    name = ['deltax', 'deltay', 'deltavx', 'deltavy']
    x_range = [(-40,40), (-1,1)]
    labels = ['$\Delta x$', '$\Delta y$', '$\Delta v_x$', '$\Delta v_y$']
    for i in range(4):
        fig, ((ax1), (ax1r)) = plt.subplots(2, 1, sharex=False, gridspec_kw={'height_ratios': [3, 1]})
        ax1.hist([limit_up[:, i], limit_down[:, i]],
                     bins=nbins,
                     density=False,
                     range=x_range[i // 2],
                     histtype='step',
                     color=['plum', 'peru'],
                     label=['generated (r=14mm)', 'generated (r=10mm)'])
        ns, bins, patches = ax1.hist([real[:,i],fake[:, i]],
                                         bins=nbins,
                                         density=False,
                                         range=x_range[i//2],
                                         histtype='step',
                                         color=['black','red'],
                                         label=['data','generated'])
        ax1.set_yscale('log')
        ax1.set_xlim(left=x_range[i // 2][0], right=x_range[i // 2][1])
        ax1.legend()

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
        ax1r.axhline(y=1, linestyle='dashed', color='black')
        ax1r.errorbar(x=b, y=r, yerr=r_yerror, xerr=r_xerror, fmt='o', color='k', ecolor='k')
        ax1r.set_xlim(left=x_range[i // 2][0], right=x_range[i // 2][1])
        ax1r.set_ylim(bottom=0, top=2)
        ax1r.set_xlabel(labels[i])
        plt.savefig(f'interpolation_ratio_{name[i]}.png', format='png')
        plt.savefig(f'interpolation_ratio_{name[i]}.pdf', format='pdf')


def compute_WD(real, fake):
    print('Computing Wasserstein distance')
    WD_matrix = np.zeros([4,8,8])
    for k in range(4):
        sumWD = 0
        for i in range(8):
            for j in range(8):
                if i == j:
                    WD_matrix[k, i, j] = wasserstein_distance(real[300000 * i:300000 * (i + 1), k],
                                                              fake[300000 * j:300000 * (j + 1), k])
                    #print(f'Var {k}: WD ({i}) = {WD_matrix[k, i, j]}')
                    sumWD += WD_matrix[k, i, j]
               # else:
               #     WD_matrix[k, i, j] = wasserstein_distance(real[300000*i:300000*(i+1),k],
               #                                               real[300000*j:300000*(j+1),k])
        print(f'MEAN WD {k} = {sumWD/8}')

    return WD_matrix


def plot_WD(WD_matrix):
    plt.rcParams["figure.figsize"] = (7, 7)
    plt.rcParams["figure.titlesize"], plt.rcParams["axes.titlesize"] = (20, 20)
    plt.rcParams["axes.labelsize"] = 18
    plt.xlabel('xlabel', fontsize=18)
    plt.ylabel('ylabel', fontsize=18)

    #fig.suptitle('Wasserstein distance between distributions', fontsize=20)
    name = ['deltax', 'deltay', 'deltavx', 'deltavy']
    labels = ['$\Delta x$', '$\Delta y$', '$\Delta v_x$', '$\Delta v_y$']
    for k in range(4):
        fig, (ax1) = plt.subplots(1, 1, sharex=False)
        a = ax1.matshow(WD_matrix[k]/np.max(WD_matrix[k]), cmap=plt.cm.Blues)
        for i in range(8):
            for j in range(8):
                if i == j:
                    c = np.around(WD_matrix[k, i, j]/np.max(WD_matrix[k]), 3)
                    ax1.text(i, j, str(c), va='center', ha='center')
        ax1.set_xlabel(labels[k])
        ax1.set_xticklabels([2,4,6,8,10,14,16,18,20])
        ax1.set_yticklabels([2,4,6,8,10,14,16,18,20])
        ax1.tick_params(axis='both', which='major', labelsize=12)
        fig.colorbar(a, location = 'bottom', fraction = 0.05)
        plt.savefig(f'WD_{name[k]}.png', format='png')
        plt.savefig(f'WD_{name[k]}.pdf', format='pdf')


def compute_KS(real, fake):
    print('Computing K-S test')
    KS_matrix = np.zeros([4,8,8])
    for k in range(4):
        sumKS = 0
        for i in range(8):
            for j in range(8):
                if i == j:
                    real_dist = real[300000 * i:300000 * (i + 1), k]
                    real_dist = real_dist[np.abs(real_dist)<1]
                    fake_dist = fake[300000 * i:300000 * (i + 1), k]
                    fake_dist = fake_dist[np.abs(fake_dist)<1]
                    print(np.size(fake_dist),np.size(real_dist))
                    res = kstest(real_dist,
                                 fake_dist)
                    print(res)
                    KS_matrix[k, i, j] = res.pvalue
                    sumKS += KS_matrix[k, i, j]
                    #print(f'Var {k}: p-value ({i},{j}) = {KS_matrix[k, i, j]}')
                #else:
                #    res = kstest(real[300000 * i:300000 * (i + 1), k],
                #                 fake[300000 * j:300000 * (j + 1), k])
                #    KS_matrix[k, i, j] = res.pvalue
        print(f'MEAN KS {k} = {sumKS/8}')

    return KS_matrix


def plot_KS(KS_matrix):
    plt.rcParams["figure.figsize"] = (7, 7)
    plt.rcParams["figure.titlesize"], plt.rcParams["axes.titlesize"] = (20, 20)
    plt.rcParams["axes.labelsize"] = 18
    plt.xlabel('xlabel', fontsize=18)
    plt.ylabel('ylabel', fontsize=18)

    #fig.suptitle('Wasserstein distance between distributions', fontsize=20)
    name = ['deltax', 'deltay', 'deltavx', 'deltavy']
    labels = ['$\Delta x$', '$\Delta y$', '$\Delta v_x$', '$\Delta v_y$']
    for k in range(4):
        fig, (ax1) = plt.subplots(1, 1, sharex=False)
        a = ax1.matshow(KS_matrix[k]/np.max(KS_matrix[k]), cmap=plt.cm.Blues)
        for i in range(8):
            for j in range(8):
                if i == j:
                    c = np.around(KS_matrix[k, i, j], 3)
                    ax1.text(i, j, str(c), va='center', ha='center')
        ax1.set_xlabel(labels[k])
        ax1.set_xticklabels([2,4,6,8,10,14,16,18,20])
        ax1.set_yticklabels([2,4,6,8,10,14,16,18,20])
        ax1.tick_params(axis='both', which='major', labelsize=12)
        fig.colorbar(a, location = 'bottom', fraction = 0.05)
        plt.savefig(f'KS_{name[k]}.png', format='png')
        plt.savefig(f'KS_{name[k]}.pdf', format='pdf')


def compute_CC(real, fake):
    print('Computing Cucconi test')
    CC_matrix = np.zeros([4,8,8])
    for k in range(4):
        sumCC = 0
        for i in range(8):
            for j in range(8):
                if i == j:
                    res = cucconi_test(real[300000 * i:300000 * (i + 1), k],
                                 fake[300000 * j:300000 * (j + 1), k],
                                 replications = 100, n_jobs = 16)
                    print(res)
                    CC_matrix[k, i, j] = res.pvalue
                    sumCC += CC_matrix[k, i, j]
                    #print(f'Var {k}: p-value ({i},{j}) = {KS_matrix[k, i, j]}')
                #else:
                #    res = kstest(real[300000 * i:300000 * (i + 1), k],
                #                 fake[300000 * j:300000 * (j + 1), k])
                #    KS_matrix[k, i, j] = res.pvalue
        print(f'MEAN Cucconi {k} = {sumCC/8}')

    return CC_matrix


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
    plt.rcParams["figure.figsize"] = (7, 7)
    plt.rcParams["figure.titlesize"], plt.rcParams["axes.titlesize"] = (20, 20)
    plt.rcParams["axes.labelsize"] = 18

    name = ['deltaxvx', 'deltayvy']
    labels = ['$|\Delta x - \Delta v_x|$ (normalized)', '$|\Delta y - \Delta v_y|$ (normalized)']
    cov_dxdvx = abs(cov_dxdvx) / np.max(abs(cov_dxdvx))
    cov_dydvy = abs(cov_dydvy) / np.max(abs(cov_dydvy))
        
    fig, (ax1) = plt.subplots(1, 1, sharex=False)
    ax1.matshow(cov_dxdvx[:,:8], cmap=plt.cm.Greens)
    ax1.set_xlabel(labels[0])
    ax1.set_xticklabels([2,4,6,8,10,14,16,18,20])
    ax1.set_yticklabels([2,4,6,8,10,14,16,18,20])
    fig.colorbar(a, location = 'bottom', fraction = 0.05)
    plt.savefig(f'cov_{name[0]}.png', format='png')
    plt.savefig(f'cov_{name[0]}.pdf', format='pdf')
    fig, (ax2) = plt.subplots(1, 1, sharex=False)
    ax2.matshow(cov_dxdvx[:,:8], cmap=plt.cm.Greens)
    ax2.set_xlabel(labels[1])
    ax2.set_xticklabels([2,4,6,8,10,14,16,18,20])
    ax2.set_yticklabels([2,4,6,8,10,14,16,18,20])
    ax2.matshow(cov_dydvy[:,:8], cmap=plt.cm.Greens)
    fig.colorbar(a, location = 'bottom', fraction = 0.05)
    plt.savefig(f'cov_{name[1]}.png', format='png')
    plt.savefig(f'cov_{name[1]}.pdf', format='pdf')



if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # GPU memory usage configuration
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

    print('Loading data...')
    real_dataset = np.load('baseEvalRealData.npy')
    fake_dataset = np.load('baseEvalGenData.npy')
    #fake_dataset = np.load('baseEvalRealData_seed19.npy')
    real_inter = np.load('baseEvalRealInterpol.npy')
    fake_inter = np.load('baseEvalGenInterpol.npy')
    labels = np.load('baseEvalLabels.npy')
    
    KS_matrix = compute_KS(real_dataset, fake_dataset)
    WD_matrix = compute_WD(real_dataset, fake_dataset)
    #CC_matrix = compute_CC(real_dataset, fake_dataset)

    NUMERICAL = False
    if NUMERICAL:
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

        WD_matrix = compute_WD(real_dataset, fake_dataset)
        KS_matrix = compute_KS(real_dataset, fake_dataset)
        cov_dxdvx, cov_dydvy = compute_covariances(real_dataset, fake_dataset)

    print('Plotting results...')
    #out = PdfPages(OUTPUT_FILE)
    '''num_classes = labels.shape[1]
    plot_difference(real_dataset,
                               fake_dataset,
                               nbins=100,
                               title='')
    plot_interpolation(real_inter,
                                  fake_inter,
                                  limit_down=fake_dataset[labels.argmax(1) == 3],
                                  limit_up=fake_dataset[labels.argmax(1) == 4],
                                  nbins=100,
                                  title='r = 12mm')
    '''
    #plot_KS(KS_matrix)
    #plot_covs(cov_dxdvx, cov_dydvy)
    #out.close()'''
    print('Evaluation done!')
