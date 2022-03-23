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
def plot_positions():
    plt.rcParams["figure.figsize"] = (14,14)
    plt.rcParams["figure.titlesize"], plt.rcParams["axes.titlesize"] = (20,20)
    plt.rcParams["axes.labelsize"] = 18
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)

    xrange = (-50,50)
    ax1.hist(x1, bins=100, density = True, histtype = 'step', color = 'black', range = xrange, label = 'data x1');
    ax1.hist(x1g, bins=100, density = True, histtype = 'step', color = 'red', range = xrange, label = 'generated x1');
    ax2.hist(y1, bins=100, density = True, histtype = 'step', color = 'black', range = xrange, label = 'data y1');
    ax2.hist(y1g, bins=100, density = True, histtype = 'step', color = 'red', range = xrange, label = 'generated y1');
    ax3.hist(x2, bins=100, density = True, histtype = 'step', color = 'black', range = xrange, label = 'data x2');
    ax3.hist(x2g, bins=100, density = True, histtype = 'step', color = 'red', range = xrange, label = 'generated x2');
    ax4.hist(y2, bins=100, density = True, histtype = 'step', color = 'black', range = xrange, label = 'data y2');
    ax4.hist(y2g, bins=100, density = True, histtype = 'step', color = 'red', range = xrange, label = 'generated y2');
    ax1.set_xlabel('$x_1$')
    ax2.set_xlabel('$y_1$')
    ax3.set_xlabel('$x_2$')
    ax4.set_xlabel('$y_2$')
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    fig.suptitle('Positions');
    return fig
    
def plot_directions(scale):
    plt.rcParams["figure.figsize"] = (14,14)
    plt.rcParams["figure.titlesize"], plt.rcParams["axes.titlesize"] = (20,20)
    plt.rcParams["axes.labelsize"] = 18
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)

    xrange = (-1.5, 1.5)
    ax1.hist(vx1, bins=100, density = True, histtype = 'step', color = 'black', range = xrange, label = 'data vx1');
    ax1.hist(vx1g, bins=100, density = True, histtype = 'step', color = 'red', range = xrange, label = 'generated vx1');
    ax2.hist(vy1, bins=100, density = True, histtype = 'step', color = 'black', range = xrange, label = 'data vy1');
    ax2.hist(vy1g, bins=100, density = True, histtype = 'step', color = 'red', range = xrange, label = 'generated vy1');
    ax3.hist(vx2, bins=100, density = True, histtype = 'step', color = 'black', range = xrange, label = 'data vx2');
    ax3.hist(vx2g, bins=100, density = True, histtype = 'step', color = 'red', range = xrange, label = 'generated vx2');
    ax4.hist(vy2, bins=100, density = True, histtype = 'step', color = 'black', range = xrange, label = 'data vy2');
    ax4.hist(vy2g, bins=100, density = True, histtype = 'step', color = 'red', range = xrange, label = 'generated vy2');
    ax1.set_xlabel('$v_{x_1}$')
    ax2.set_xlabel('$v_{y_1}$')
    ax3.set_xlabel('$v_{x_2}$')
    ax4.set_xlabel('$v_{y_2}$')
    ax1.set_yscale(scale)
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    fig.suptitle('Velocities');
    return fig

def plot_detector(title, first):
    plt.rcParams["figure.figsize"] = (14,14)
    plt.rcParams["figure.titlesize"], plt.rcParams["axes.titlesize"] = (20,20)
    plt.rcParams["axes.labelsize"] = 18
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey=True)

    xrange = (-1,1)
    if first:
        hx1 = ax1.hist(vx1, bins=nbins, density = True, histtype = 'step', color = 'black', range = xrange, label = 'data');
        hy1 = ax2.hist(vy1, bins=nbins, density = True, histtype = 'step', color = 'red', range = xrange, label = 'data');
        if options.fit:
            popt, pcov = curve_fit(cos2, np.linspace(*xrange, nbins), hx1[0])
            ax1.plot(np.linspace(*xrange, nbins), cos2(np.linspace(*xrange, nbins), *popt), label=f'$Acos(bx)^2$ fit: \n [{popt[0]:0.2e}, {popt[1]:0.2e}]', color = 'black')
            popt, pcov = curve_fit(cos2, np.linspace(*xrange, nbins), hy1[0])
            ax2.plot(np.linspace(*xrange, nbins), cos2(np.linspace(*xrange, nbins), *popt), label=f'$Acos(bx)^2$ fit: \n [{popt[0]:0.2e}, {popt[1]:0.2e}]', color = 'red')
        ax3.hist(thetax1, bins=nbins, density = True, range = (0,0.8), histtype = 'step', color = 'black')
        ax4.hist(thetay1, bins=nbins, density = True, range = (0,0.8), histtype = 'step', color = 'red')
        ax3.set_xlabel('$\Theta_{x_1}$')
        ax4.set_xlabel('$\Theta_{y_1}$')
        ax1.set_xlabel('$v_{x_1}$')
        ax2.set_xlabel('$v_{y_1}$')
    else:
        hx1 = ax1.hist(vx2, bins=nbins, density = True, histtype = 'step', color = 'black', range = xrange, label = 'data');
        hy1 = ax2.hist(vy2, bins=nbins, density = True, histtype = 'step', color = 'red', range = xrange, label = 'data');
        if options.fit:
            popt, pcov = curve_fit(cos2, np.linspace(*xrange, nbins), hx1[0])
            ax1.plot(np.linspace(*xrange, nbins), cos2(np.linspace(*xrange, nbins), *popt), label=f'$Acos(bx)^2$ fit: \n [{popt[0]:0.2e}, {popt[1]:0.2e}]', color = 'black')
            popt, pcov = curve_fit(cos2, np.linspace(*xrange, nbins), hy1[0])
            ax2.plot(np.linspace(*xrange, nbins), cos2(np.linspace(*xrange, nbins), *popt), label=f'$Acos(bx)^2$ fit: \n [{popt[0]:0.2e}, {popt[1]:0.2e}]', color = 'red')
        ax3.hist(thetax2, bins=nbins, density = True, range = (0,0.8), histtype = 'step', color = 'black')
        ax4.hist(thetay2, bins=nbins, density = True, range = (0,0.8), histtype = 'step', color = 'red')
        ax3.set_xlabel('$\Theta_{x_2}$')
        ax4.set_xlabel('$\Theta_{y_2}$')
        ax1.set_xlabel('$v_{x_2}$')
        ax2.set_xlabel('$v_{y_2}$') 
    ax1.legend()
    ax2.legend()
    fig.suptitle(title);
    return fig

def plot_vector_difference(scale):
    plt.rcParams["figure.figsize"] = (14,7)

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    xrange = (-0.3,0.3)

    ax1.hist(vx2-vx1, bins=100, density = True, range = xrange, histtype = 'step', color = 'black', log = False, label = 'data $\Delta v_x$')
    ax1.hist(vx2g-vx1g, bins=100, density = True, range = xrange, histtype = 'step', color = 'red', log = False, label = 'generated $\Delta v_x$')
    ax1.set_xlabel('$\Delta v_x$')
    ax1.set_yscale(scale)
    ax1.legend()
    ax2.hist(vy2-vy1, bins=100, density = True, range = xrange, histtype = 'step', color = 'black', log = False, label = 'data $\Delta v_y$')
    ax2.hist(vy2g-vy1g, bins=100, density = True, range = xrange, histtype = 'step', color = 'red', log = False, label = 'generated $\Delta v_x$')
    ax2.set_xlabel('$\Delta v_y$')
    ax2.set_yscale(scale)
    ax2.legend()
    fig.suptitle('Vector difference');
    return fig

def plot_difference(var1, var2, var1g, var2g, x_range, labels, scale):
    plt.rcParams["figure.figsize"] = (14,7)
    xrange = (-5,5)

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)

    h1 = ax1.hist(var1,  bins=100, density = True, range = x_range, histtype = 'step', color = 'black', label='data '+labels[0])
    h1 = ax1.hist(var1g, bins=100, density = True, range = x_range, histtype = 'step', color = 'red',   label='generated '+labels[0])
    h2 = ax2.hist(var2,  bins=100, density = True, range = x_range, histtype = 'step', color = 'black', label='data '+labels[1])
    h2 = ax2.hist(var2g, bins=100, density = True, range = x_range, histtype = 'step', color = 'red',   label='generated '+labels[1])
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


if __name__ == '__main__':

	########################################################################################################################################
	######################## Parse options  ################################################################################################
	########################################################################################################################################
	parser = OptionParser()
	parser.add_option('-p', '--predictions', action='store_true', dest='mode')
	parser.add_option("-d", "--in-data", dest="in_data", type="string", default='input.root', help="Input root file")
	parser.add_option("-g", "--in-gen", dest="in_gen", type="string", default='input.root', help="Input root file")
	parser.add_option("-b", "--bins", dest="nbins", type="int", default='100', help="Number of bins in histogram")
	(options, args) = parser.parse_args()

	########################################################################################################################################
	######################## Read File      ################################################################################################
	########################################################################################################################################
	filename = options.in_data
	f = ROOT.TFile(filename, 'READ')
	data = ROOT.RDataFrame('globalReco', f).AsNumpy()
	filename = options.in_gen
	f = ROOT.TFile(filename, 'READ')
	gen = ROOT.RDataFrame('globalReco', f).AsNumpy()
	filename = filename.replace('.root','')
	nbins = options.nbins

	cut_positions(data, gen)

	# Rename variables
	if options.mode:
		x1, y1 = data['px1'], data['py1']
		x2, y2 = data['px2'], data['py2']
		vx1, vy1 = data['pvx1'], data['pvy1']
		vx2, vy2 = data['pvx2'], data['pvy2']
		x1g, y1g = gen['px1'], gen['py1']
		x2g, y2g = gen['px2'], gen['py2']
		vx1g, vy1g = gen['pvx1'], gen['pvy1']
		vx2g, vy2g = gen['pvx2'], gen['pvy2']
	else:
		x1, y1, = data['mx1'], data['my1']
		x2, y2, = data['mx2'], data['my2']
		vx1, vy1 = data['mvx1'], data['mvy1']
		vx2, vy2 = data['mvx2'], data['mvy2']
		x1g, y1g = gen['mx1'], gen['my1']
		x2g, y2g = gen['mx2'], gen['my2']
		vx1g, vy1g = gen['mvx1'], gen['mvy1']
		vx2g, vy2g = gen['mvx2'], gen['mvy2']

	########################################################################################################################################
	######################## Calculate variables     #######################################################################################
	########################################################################################################################################
	# Calculate angular distributions
	thetax1 = np.arccos(1/np.sqrt(vx1**2+1))
	thetay1 = np.arccos(1/np.sqrt(vy1**2+1))
	thetax2 = np.arccos(1/np.sqrt(vx2**2+1))
	thetay2 = np.arccos(1/np.sqrt(vy2**2+1))
	Dthetax = thetax2-thetax1
	Dthetay = thetay2-thetay1

	# Calculate position difference
	l = -39*2
	#  data
	x = x1 + l*vx1
	y = y1 + l*vy1
	Dx = x2 - x
	Dy = y2 - y
	#  generated
	x = x1g + l*vx1g
	y = y1g + l*vy1g
	Dxg = x2g - x
	Dyg = y2g - y

	# Print pdf file
	if options.mode:
		out = PdfPages(filename+'_predictions.pdf')
	else:
		out = PdfPages(filename+'_measurements.pdf')
	out.savefig(plot_positions());
	out.savefig(plot_directions('log'));
	#out.savefig(plot_detector('First detector', True));
	#out.savefig(plot_detector('Second detector', False));
	out.savefig(plot_difference(Dx, Dy, Dxg, Dyg, (-5,5), ['$\Delta x','$\Delta y'], 'log'));
	out.savefig(plot_difference(vx2-vx1, vy2-vy1, vxg2-vxg1, vyg2-vyg1, (-1,1), ['$\Delta v_x','$\Delta v_y'], 'log'));
	out.savefig(plot_correlation_2Dhist(x2, vx2, x2g, vx2g, [[-50,50],[-1,1]], ['$x_2$','$v_{x_2}$']));
	out.savefig(plot_correlation_2Dhist(y2, vy2, y2g, vy2g, [[-50,50],[-1,1]], ['$y_2$','$v_{y_2}$']));
	out.savefig(plot_correlation_2Dhist(Dx, vx2-vx1, Dxg, vx2g-vx1g, [[-5,5],[-0.5,0.5]], ['$\Delta x$','$\Delta v_x$']));
	out.savefig(plot_correlation_2Dhist(Dy, vy2-vy1, Dyg, vy2g-vy1g, [[-5,5],[-0.5,0.5]], ['$\Delta y$','$\Delta v_y$']));
	out.close()
