##################################################################################################
#### Plotting module                                                                          ####
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
######################## Plotting functions  ###########################################################################################
########################################################################################################################################
def plot_hits(hits_real, hits_gen):
    plt.rcParams["figure.figsize"] = (14,7)
    plt.rcParams["figure.titlesize"], plt.rcParams["axes.titlesize"] = (20,20)
    plt.rcParams["axes.labelsize"] = 18
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    
    ax1.hist(hits_real, density = False, histtype = 'step', color = 'black', range=(0,6), bins=6, label='Real')
    ax2.hist(hits_gen, density = False, histtype = 'step', color = 'red', range=(0,6), bins=6, label='Generated')
    ax1.legend()
    ax2.legend()
    fig.suptitle('Number of hits');
    return fig

def plot_wire_pos(activations, activations_gen):
    plt.rcParams["figure.figsize"] = (14,7)
    plt.rcParams["figure.titlesize"], plt.rcParams["axes.titlesize"] = (20,20)
    plt.rcParams["axes.labelsize"] = 18
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    
    ax1.hist(activations,     color = 'black', range=(0,216), bins=216, label='Real',      log=False, bottom=0)
    ax2.hist(activations_gen, color = 'red',   range=(0,216), bins=216, label='Generated', log=False, bottom=0)
    ax1.legend()
    ax2.legend()
    fig.suptitle('Number of hits');
    return fig


def print_plots(activations, activations_gen):
    plot_wire_pos(activations, activations_gen)
