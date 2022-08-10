import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def load(inputfile):
    data = pd.read_csv(inputfile).to_numpy()
    # Select only some radius
    mask = [i in [4,6,8,10,12,14,16,18,20] for i in data[:,8]]
    data = data[mask]

    variables = data[:,:8]
    labels = data[:,8]

    return variables, labels

def plot_variable(var, labels_r, xrange, label, ind=True):
    plt.rcParams["figure.figsize"] = (7, 7)
    plt.rcParams["figure.titlesize"], plt.rcParams["axes.titlesize"] = (30, 30)
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams["xtick.labelsize"], plt.rcParams["ytick.labelsize"] = (14, 14)

    fig, ax = plt.subplots()
    if ind:
        custom_colors = ['#54ac68', '#b04e0f', '#fc2647', '#fffe40', '#6140ef', '#ff69af', '#feb308', '#155084', '#a8a495']
        for k in range(9):
            i = np.unique(labels_r)[k]
            legend_label = 'r = ' + str(int(i))
            ax.hist(var[labels_r == i],
                    bins=200,
                    range=xrange,
                    label=legend_label,
                    density=True,
                    histtype='step',
                    color=custom_colors[k])
            ax.legend(prop={'size': 12})
    else:
        ax.hist(var,
                bins=200,
                range=xrange,
                density=True,
                histtype='step',
                color='black')
    ax.set_yscale('log')
    ax.set_xlabel(label)
    return fig


if __name__=='__main__':
    LOAD = False

    if LOAD:
        print('Loading data...')
        inputfile = '~/fewSamples/training_samples.csv'
        variables, labels_r = load(inputfile)

    print('Plotting variables...')
    output = 'final_plots'
    out = PdfPages('/home/ruben/' + output + '.pdf')
    ranges = [(-1.5,1.5), (-40,40)]
    labels = ['$x_1$', '$y_1$', '$v_{x_1}$', '$v_{y_1}$',
              '$\Delta x^*$', '$\Delta y^*$', '$\Delta v_x$', '$\Delta v_y$']
    for j in range(8):
        if j % 4 in [2,3]:
            xrange = ranges[0]
        else:
            xrange = ranges[1]
        if j in [0,1,2,3]:
            ind = False
        else:
            ind = True
        out.savefig(plot_variable(var=variables[:,j],
                                  labels_r=labels_r,
                                  xrange=xrange,
                                  label=labels[j],
                                  ind=ind))
        plt.show()
    out.close()
    print('Done')