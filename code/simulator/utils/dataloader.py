##################################################################################################
#### Data loading module                                                                      ####
##################################################################################################
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load(inputfile):
    '''
    Reads a .csv file containin the training data.
    
    Parameters
    ----------
    inputfile : string
        Path to csv file

    Returns
    -------
    input_data : NumPy Array of shape (n_events,2)
        Array with the exact positions and velocities of the muons.
    activations : NumPy Array of shape (n_events,n_wires=216)
        Matrix containing the activation of the different wires (1 if activated, 0 if not).
    '''
    data = pd.read_csv(inputfile).to_numpy()
    hits = np.sum(data[:,2:], axis=1)
    mask = hits<5
    data = data[mask,:]
    print(data.shape)
    input_data = data[:,0:2] # [px, theta]
    # convert px to centimeters and theta to velocities (cos(theta))___
    input_data[:,0] = input_data[:,0]*0.4
    input_data[:,1] = np.cos(input_data[:,1])
    #__________________________________________________________________
    activations = data[:,2:]
    return [input_data, activations]

def scale(dataset):
    in_data, activations = dataset
    scaler = StandardScaler()
    scaler.fit(in_data)
    in_data_scaled = scaler.transform(in_data)
	
	# compute weights
    w = None
    return [in_data_scaled, activations], w, scaler