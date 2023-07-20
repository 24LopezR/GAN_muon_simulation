##################################################################################################
#### Data loading module                                                                      ####
##################################################################################################
import numpy as np
import ROOT as r


def load(inputfile, detector=1):
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
    if detector == 1:
        for ev in f.globalReco:
            if ev.type1 != 3 or ev.type2 != 3:
                continue
            if abs(ev.px1) > 80 or abs(ev.py1) > 80 or abs(ev.pvx1) > 1.5 or abs(ev.pvy1) > 1.5:
                continue
            if abs(ev.mx1) > 80 or abs(ev.my1) > 80 or abs(ev.mvx1) > 1.5 or abs(ev.mvy1) > 1.5:
                continue
            thedata.append([ev.px1, ev.py1, ev.pvx1, ev.pvy1, ev.mx1, ev.my1, ev.mvx1, ev.mvy1])
    if detector == 2:
        for ev in f.globalReco:
            if ev.type1 != 3 or ev.type2 != 3:
                continue
            if abs(ev.px2) > 80 or abs(ev.py2) > 80 or abs(ev.pvx2) > 1.5 or abs(ev.pvy2) > 1.5:
                continue
            if abs(ev.mx2) > 80 or abs(ev.my2) > 80 or abs(ev.mvx2) > 1.5 or abs(ev.mvy2) > 1.5:
                continue
            thedata.append([ev.px2, ev.py2, ev.pvx2, ev.pvy2, ev.mx2, ev.my2, ev.mvx2, ev.mvy2])
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
		dataset = [m_vars, p_vars]: numpy array of shape [(N, 4), (N, 4)]
		w: computed weights, numpy array of size (N, 1)
	"""
	scaler.fit(data)
	data_transf = scaler.transform(data)
	m_vars = data_transf[:,:4]
	p_vars = data_transf[:,4:]
	
	# compute weights
	w = None
	return [m_vars, p_vars], w, scaler