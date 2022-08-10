##################################################################################################
#### Plotting module                                                                          ####
##################################################################################################
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

########################################################################################################################################
######################## Fit functions  ################################################################################################
########################################################################################################################################
def gauss_function(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))
def cos2(x, a, b):
    y = np.zeros(x.size)
    i = 0
    for x in x:
        y[i] = np.piecewise(x, [x<-np.pi/2/b, x>-np.pi/2/b and x<np.pi/2/b, x>np.pi/2/b], [np.zeros(x.size), a*np.cos(b*(x))**2, np.zeros(x.size)])
        i = i+1
    return y


########################################################################################################################################
######################## Plotting functions  ###########################################################################################
########################################################################################################################################
def plot_positions(var1, var2, var1g, var2g, xrange, scale):
    plt.rcParams["figure.figsize"] = (14,7)
    plt.rcParams["figure.titlesize"], plt.rcParams["axes.titlesize"] = (20,20)
    plt.rcParams["axes.labelsize"] = 18
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)

    b = 200
    ax1.hist(var1, bins=b, density = False, histtype = 'step', color = 'black', range = xrange, label = 'data x2');
    if var1g is not None:
        ax1.hist(var1g, bins=b, density = False, histtype = 'step', color = 'red', range = xrange, label = 'generated x2');
    ax2.hist(var2, bins=b, density = False, histtype = 'step', color = 'black', range = xrange, label = 'data y2');
    if var2g is not None:
        ax2.hist(var2g, bins=b, density = False, histtype = 'step', color = 'red', range = xrange, label = 'generated y2');
    ax1.set_xlabel('$x_2$')
    ax2.set_xlabel('$y_2$')
    ax1.set_yscale(scale)
    ax1.legend()
    ax2.legend()
    fig.suptitle('Positions');
    return fig
    
def plot_directions(var1, var2, var1g, var2g, xrange, scale):
    plt.rcParams["figure.figsize"] = (14,7)
    plt.rcParams["figure.titlesize"], plt.rcParams["axes.titlesize"] = (20,20)
    plt.rcParams["axes.labelsize"] = 18
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)

    b = 200
    ax1.hist(var1, bins=b, density = False, histtype = 'step', color = 'black', range = xrange, label = 'data vx2');
    if var1g is not None:
        ax1.hist(var1g, bins=b, density = False, histtype = 'step', color = 'red', range = xrange, label = 'generated vx2');
    ax2.hist(var2, bins=b, density = False, histtype = 'step', color = 'black', range = xrange, label = 'data vy2');
    if var2g is not None:
        ax2.hist(var2g, bins=b, density = False, histtype = 'step', color = 'red', range = xrange, label = 'generated vy2');
    ax1.set_xlabel('$v_{x_2}$')
    ax2.set_xlabel('$v_{y_2}$')
    ax1.set_yscale(scale)
    ax1.legend()
    ax2.legend()
    fig.suptitle('Velocities');
    return fig

def plot_difference(var1, var2, var1g, var2g, x_range, labels, scale, bins=200):
    plt.rcParams["figure.figsize"] = (14,7)
    plt.rcParams["figure.titlesize"], plt.rcParams["axes.titlesize"] = (20,20)
    plt.rcParams["axes.labelsize"] = 18

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    b = bins
    ax1.hist(var1,  bins=b, density = False, range = x_range, histtype = 'step', color = 'black', label='data '+labels[0]);
    if var1g is not None:
        ax1.hist(var1g, bins=b, density = False, range = x_range, histtype = 'step', color = 'red',   label='generated '+labels[0]);
    ax2.hist(var2,  bins=b, density = False, range = x_range, histtype = 'step', color = 'black', label='data '+labels[1]);
    if var2g is not None:
        ax2.hist(var2g, bins=b, density = False, range = x_range, histtype = 'step', color = 'red',   label='generated '+labels[1]);
    ax1.set_xlabel(labels[0])
    ax2.set_xlabel(labels[1])
    ax1.set_yscale(scale)
    ax2.set_yscale(scale)
    ax1.legend()
    ax2.legend()
    return fig

def plot_correlation_2Dhist(var1, var2, var1g, var2g, ranges, labels):
    plt.rcParams["figure.figsize"] = (14,7)
    plt.rcParams["figure.titlesize"], plt.rcParams["axes.titlesize"] = (20,20)
    plt.rcParams["axes.labelsize"] = 18

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)

    ax1.hist2d(var1, var2, bins=200, range = ranges, density = True,
              norm=matplotlib.colors.LogNorm())
    ax1.set_xlabel(labels[0])
    ax1.set_ylabel(labels[1])
    if var1g is not None:
        ax2.hist2d(var1g, var2g, bins=200, range = ranges, density = True,
              norm=matplotlib.colors.LogNorm())
        ax2.set_xlabel(labels[0])
        ax2.set_ylabel(labels[1])
    #fig.suptitle('Correlation '+labels[0]+'-'+labels[1]);
    return fig


def print_plots(data, fake, output):

    if fake is not None:
        out = PdfPages('evaluation_'+output+'.pdf')
        out.savefig(plot_difference(data[:, 4], data[:, 5], fake[:, 0],
                    fake[:, 1], (-25, 25), ['$\Delta x$', '$\Delta y$'], 'log'))
        out.savefig(plot_difference(data[:, 6], data[:, 7], fake[:, 2],
                    fake[:, 3], (-1.5, 1.5), ['$\Delta v_x$', '$\Delta v_y$'], 'log'))
        out.savefig(plot_correlation_2Dhist(data[:, 4], data[:, 6], fake[:, 0], fake[:, 2], [
                    [-15, 15], [-0.5, 0.5]], ['$\Delta x$', '$\Delta v_x$']))
        out.savefig(plot_correlation_2Dhist(data[:, 5], data[:, 7], fake[:, 1], fake[:, 3], [
                    [-15, 15], [-0.5, 0.5]], ['$\Delta y$', '$\Delta v_y$']))

        # Plot variables x2, y2, vx2, vy2
        l = -39*2
        x1 = data[:, 0]
        y1 = data[:, 1]
        vx1 = data[:, 2]
        vy1 = data[:, 3]
        x2_real = data[:, 4] + x1 - l*vx1
        y2_real = data[:, 5] + y1 - l*vy1
        vx2_real = data[:, 6] + vx1
        vy2_real = data[:, 7] + vy1
        x2_fake = fake[:, 0] + x1 - l*vx1
        y2_fake = fake[:, 1] + y1 - l*vy1
        vx2_fake = fake[:, 2] + vx1
        vy2_fake = fake[:, 3] + vy1
        out.savefig(plot_positions(x2_real, y2_real,
                    x2_fake, y2_fake, (-80, 80), 'linear'))
        out.savefig(plot_directions(vx2_real, vy2_real,
                    vx2_fake, vy2_fake, (-1.5, 1.5), 'log'))
        out.close()
    else:
        out = PdfPages('evaluation_'+output+'.pdf')
        out.savefig(plot_difference(
            data[:, 4], data[:, 5], None, None, (-25, 25), ['$\Delta x$', '$\Delta y$'], 'log'))
        out.savefig(plot_difference(data[:, 6], data[:, 7], None,
                    None, (-1.5, 1.5), ['$\Delta v_x$', '$\Delta v_y$'], 'log'))
        out.savefig(plot_correlation_2Dhist(data[:, 4], data[:, 6], None, None, [
                    [-15, 15], [-0.5, 0.5]], ['$\Delta x$', '$\Delta v_x$']))
        out.savefig(plot_correlation_2Dhist(data[:, 5], data[:, 7], None, None, [
                    [-15, 15], [-0.5, 0.5]], ['$\Delta y$', '$\Delta v_y$']))

        # Plot variables x2, y2, vx2, vy2
        l = -39*2
        x1 = data[:, 0]
        y1 = data[:, 1]
        vx1 = data[:, 2]
        vy1 = data[:, 3]
        x2_real = data[:, 4] + x1 - l*vx1
        y2_real = data[:, 5] + y1 - l*vy1
        vx2_real = data[:, 6] + vx1
        vy2_real = data[:, 7] + vy1
        out.savefig(plot_positions(x2_real, y2_real,
                    None, None, (-60, 60), 'linear'))
        out.savefig(plot_directions(vx2_real, vy2_real,
                    None, None, (-10, 10), 'log'))
        out.close()
