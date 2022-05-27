import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def plot_difference(var_real, var_fake, x_range, label, scale='log', bins=100):
    plt.rcParams["figure.figsize"] = (14, 14)
    plt.rcParams["figure.titlesize"], plt.rcParams["axes.titlesize"] = (20, 20)
    plt.rcParams["axes.labelsize"] = 18

    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
    b = bins
    n_real, bins_real, _ = ax1.hist(var_real,
                                     bins=b,
                                     density=False,
                                     range=x_range,
                                     histtype='step',
                                     color='black',
                                     alpha=1,
                                     label='data')
    n_fake, bins_fake, _ = ax1.hist(var_fake,
                                     bins=b,
                                     density=False,
                                     range=x_range,
                                     histtype='step',
                                     color='red',
                                     alpha=1,
                                     label='generated')

    ratio = []
    for i in range(n_real.size):
        if n_real[i] == 0:
            ratio.append(0)
        else:
            ratio.append(n_fake[i]/n_real[i])
    ratio = np.asarray(ratio)

    ax2.scatter(bins_real[:-1], ratio, c='black', marker='.')

    ax1.set_yscale(scale)
    ax1.legend()
    ax2.set_yscale('linear')
    fig.suptitle(label)
    return fig


def plot_correlation_2Dhist(var1, var2, var1g, var2g, ranges, labels):
    plt.rcParams["figure.figsize"] = (14, 7)
    plt.rcParams["figure.titlesize"], plt.rcParams["axes.titlesize"] = (20, 20)
    plt.rcParams["axes.labelsize"] = 18

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)

    ax1.hist2d(var1, var2, bins=200, range=ranges, density=True,
               norm=matplotlib.colors.LogNorm())
    ax1.set_xlabel(labels[0])
    ax1.set_ylabel(labels[1])
    if var1g is not None:
        ax2.hist2d(var1g, var2g, bins=200, range=ranges, density=True,
                   norm=matplotlib.colors.LogNorm())
        ax2.set_xlabel(labels[0])
        ax2.set_ylabel(labels[1])
    fig.suptitle('Correlation ' + labels[0] + '-' + labels[1])
    return fig
