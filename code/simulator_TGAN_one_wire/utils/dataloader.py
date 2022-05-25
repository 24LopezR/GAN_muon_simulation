##################################################################################################
#### Data loading module                                                                      ####
##################################################################################################
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from textwrap import wrap

MAX = 2
GAMMA = 1e-4

def encoding(X, max_value=1):
    nonzero = np.zeros((X.shape[0], max_value))
    for i in range(X.shape[0]):
        n = np.nonzero(X[i])[0]+1
        nonzero[i,0:n.size] = n
    return np.asarray(nonzero, dtype=int)

def one_hot_encode(X, n_values=216):
    encoded = np.zeros((X.size, n_values))
    encoded[np.arange(X.size), X] = 1
    return encoded

def efficiency(X):
    broken_wires = (23,24,56 ,187,188,215)
    efficiencies = (0, 0 ,0.3,0.2,0.3,0  )
    for i in range(6):
        p = np.random.uniform(0, 1, X.shape[0])
        for j in range(X.shape[0]): 
            if p[j] > efficiencies[i]: X[j,broken_wires[i]] = 0
    return X

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
    mask = hits<=MAX
    data = data[mask,:]
    print(data.shape)
    input_data = data[:,0:2] # [px, theta]
    # convert px to centimeters and theta to velocities (cos(theta))___
    input_data[:,0] = input_data[:,0]*0.4
    input_data[:,1] = input_data[:,1]
    #__________________________________________________________________
    activations = data[:,2:]
    # break some of the wires
    activations = efficiency(activations)
    return [input_data, activations]

def scale(dataset):
    in_data, activations = dataset
    
    # scale input data
    scaler = StandardScaler()
    scaler.fit(in_data)
    in_data_scaled = scaler.transform(in_data)
    
    # encode activations
    activations = encoding(activations, MAX)
    activations = activations[:,1]
     
    # one-hot encode activations
    #activations = np.apply_along_axis(lambda row: one_hot_encode(row, n_wires=216), axis=1, arr=activations)
    activations = np.apply_along_axis(lambda row: one_hot_encode(row), axis=0, arr=activations)
    #noise = np.random.uniform(0, GAMMA, activations.shape)
    #activations = activations + noise
    # renormalize
    #s = np.repeat(np.sum(activations, axis=1), 216, axis=0).reshape(activations.shape)
    #activations = activations/s
    
    return [in_data_scaled, activations]

#[in_data, act] = load('/home/ruben/Documents/TFM/GAN_muon_simulation/data/file_2.csv')
#[in_data, act] = scale([in_data, act])