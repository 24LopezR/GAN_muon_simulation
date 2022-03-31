##################################################################################################
#### Plotting module                                                                        ####
##################################################################################################
import ROOT
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages
import sys
from optparse import OptionParser

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
######################## Data cut function #############################################################################################
########################################################################################################################################
def cut_positions(data, gen):
    if options.mode:
        cut = (data['px2'] >= -40) & (data['px2'] <= 40) & (data['py2'] >= -40) & (data['py2'] <= 40)
        for key in data.keys():
            data[key] = data[key][cut]
        cut = (gen['px2'] >= -40) & (gen['px2'] <= 40) & (gen['py2'] >= -40) & (gen['py2'] <= 40)
        for key in gen.keys():
            gen[key] = gen[key][cut]
    else:
        cut = (data['mx2'] >= -40) & (data['mx2'] <= 40) & (data['my2'] >= -40) & (data['my2'] <= 40)
        for key in data.keys():
            data[key] = data[key][cut]
        cut = (gen['mx2'] >= -40) & (gen['mx2'] <= 40) & (gen['my2'] >= -40) & (gen['my2'] <= 40)
        for key in gen.keys():
            gen[key] = gen[key][cut]


########################################################################################################################################
######################## Plotting functions  ###########################################################################################
########################################################################################################################################
def plot_positions(var1, var2, var1g, var2g):
    plt.rcParams["figure.figsize"] = (14,7)
    plt.rcParams["figure.titlesize"], plt.rcParams["axes.titlesize"] = (20,20)
    plt.rcParams["axes.labelsize"] = 18
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)

    xrange = (-50,50)
    ax1.hist(var1, bins=100, density = True, histtype = 'step', color = 'black', range = xrange, label = 'data x2');
    ax1.hist(var1g, bins=100, density = True, histtype = 'step', color = 'red', range = xrange, label = 'generated x2');
    ax2.hist(var2, bins=100, density = True, histtype = 'step', color = 'black', range = xrange, label = 'data y2');
    ax2.hist(var2g, bins=100, density = True, histtype = 'step', color = 'red', range = xrange, label = 'generated y2');
    ax1.set_xlabel('$x_1$')
    ax2.set_xlabel('$y_1$')
    ax1.legend()
    ax2.legend()
    fig.suptitle('Positions');
    return fig
    
def plot_directions(var1, var2, var1g, var2g, scale):
    plt.rcParams["figure.figsize"] = (14,7)
    plt.rcParams["figure.titlesize"], plt.rcParams["axes.titlesize"] = (20,20)
    plt.rcParams["axes.labelsize"] = 18
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)

    xrange = (-1.5, 1.5)
    ax1.hist(var1, bins=100, density = True, histtype = 'step', color = 'black', range = xrange, label = 'data vx2');
    ax1.hist(var1g, bins=100, density = True, histtype = 'step', color = 'red', range = xrange, label = 'generated vx2');
    ax2.hist(var2, bins=100, density = True, histtype = 'step', color = 'black', range = xrange, label = 'data vy2');
    ax2.hist(var2g, bins=100, density = True, histtype = 'step', color = 'red', range = xrange, label = 'generated vy2');
    ax1.set_xlabel('$v_{x_1}$')
    ax2.set_xlabel('$v_{y_1}$')
    ax1.set_yscale(scale)
    ax1.legend()
    ax2.legend()
    fig.suptitle('Velocities');
    return fig

def plot_difference(var1, var2, var1g, var2g, x_range, labels, scale):
    plt.rcParams["figure.figsize"] = (14,7)
    xrange = (-5,5)

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    b = 150
    h1 = ax1.hist(var1,  bins=b, density = False, range = x_range, histtype = 'step', color = 'black', label='data '+labels[0])
    h1 = ax1.hist(var1g, bins=b, density = False, range = x_range, histtype = 'step', color = 'red',   label='generated '+labels[0])
    h2 = ax2.hist(var2,  bins=b, density = False, range = x_range, histtype = 'step', color = 'black', label='data '+labels[1])
    h2 = ax2.hist(var2g, bins=b, density = False, range = x_range, histtype = 'step', color = 'red',   label='generated '+labels[1])
    ax1.set_xlabel(labels[0])
    ax2.set_xlabel(labels[1])
    ax1.set_yscale(scale)
    ax2.set_yscale(scale)
    ax1.legend()
    ax2.legend()
    return fig

def plot_correlation_2Dhist(var1, var2, var1g, var2g, ranges, labels):
    plt.rcParams["figure.figsize"] = (14,7)

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)

    ax1.hist2d(var1, var2, bins=100, range = ranges, density = True,
              norm=matplotlib.colors.LogNorm())
    ax1.set_xlabel(labels[0])
    ax1.set_ylabel(labels[1])
    ax2.hist2d(var1g, var2g, bins=100, range = ranges, density = True,
              norm=matplotlib.colors.LogNorm())
    ax2.set_xlabel(labels[0])
    ax2.set_ylabel(labels[1])
    fig.suptitle('Correlation '+labels[0]+'-'+labels[1]);
    return fig


def print_plots(real, fake, output):
	
	out = PdfPages('evaluation_'+output+'.pdf')
	out.savefig(plot_difference(real[:,0], real[:,1], fake[:,0], fake[:,1], (-50,50), ['$\Delta x$','$\Delta y$'], 'log'));
	out.savefig(plot_difference(real[:,2], real[:,3], fake[:,2], fake[:,3], (-1.5,1.5), ['$\Delta v_x$','$\Delta v_y$'], 'log'));
	out.savefig(plot_correlation_2Dhist(real[:,0], real[:,2], fake[:,0], fake[:,2], [[-15,15],[-1,1]], ['$\Delta x$','$\Delta v_x$']));
	out.savefig(plot_correlation_2Dhist(real[:,1], real[:,3], fake[:,1], fake[:,3], [[-15,15],[-1,1]], ['$\Delta y$','$\Delta v_y$']));
	out.close()
