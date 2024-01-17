##################################################################################################
#### Data loading module                                                                      ####
##################################################################################################
import numpy as np
import ROOT as r


def load(inputfile):
	"""
	Reads the input file with the muon data and return the dataset as a numpy array
	Arguments:
	Input:
		inputfile: name of the .root file with the data
		scaler: (sklearn.preprocessing.StandardScaler()) 
	Output: 
		data: numpy array containing the training data
	"""
	f = r.TFile(inputfile)
	thedata = []
	for ev in f.globalReco:
	    if ev.type1 != 3 or ev.type2 != 3:
	        continue
	    if abs(ev.px1) > 80 or abs(ev.py1) > 80 or abs(ev.pvx1) > 1.5 or abs(ev.pvy1) > 1.5:
	        continue
	    if abs(ev.px2) > 80 or abs(ev.py2) > 80 or abs(ev.pvx2) > 1.5 or abs(ev.pvy2) > 1.5:
	        continue
	    thedata.append([ev.px1, ev.py1, ev.pvx1, ev.pvy1, ev.px2-ev.px1 + 39*2 *
	                   ev.pvx1, ev.py2-ev.py1 + 39*2 * ev.pvy1, ev.pvx2-ev.pvx1, ev.pvy2-ev.pvy1])
	data = np.asarray(thedata)
	return data

def scale(data, scaler):
	"""
	Scales the training dataset.
	Arguments:
	Input:
		data: numpy array containing the training data
		scaler: (sklearn.preprocessing.StandardScaler()) 
	Output: 
		dataset = [second_det, first_det]: numpy array of shape [(N, 4), (N, 4)]
		w: computed weights, numpy array of size (N, 1)
	"""
	# weight events
	w = 1/np.sqrt(data[:,4]**2+data[:,5]**2)
	scaler.fit(data, sample_weight = w)
	data_transf = scaler.transform(data)
	first_det = data_transf[:,:4]
	second_det = data_transf[:,4:]
	
	# compute weights
	w = None
	return [second_det, first_det], w, scaler