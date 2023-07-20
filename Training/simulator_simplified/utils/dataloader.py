##################################################################################################
#### Data loading module                                                                      ####
##################################################################################################
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from textwrap import wrap
from bitarray import bitarray 
from bitarray.util import ba2int

MAX = 5
GAMMA = 0.2

def encoding(X, max_value):
    nonzero = np.zeros((X.shape[0], max_value))
    for i in range(X.shape[0]):
        n = np.nonzero(X[i])[0]+1
        nonzero[i,0:n.size] = n
    return np.asarray(nonzero, dtype=int)

def one_hot_encode(X, n_values):
    one_hot_X = np.zeros(X.size*n_values)
    for i in range(X.size):
        cat = X[i]
        one_hot_X[n_values*i+cat] = 1
    return one_hot_X
    
def efficiency(X):
    broken_wires = (23,24,56 ,187,188,215)
    efficiencies = (0, 0 ,0.3,0.2,0.3,0  )
    for i in range(6):
        p = np.random.uniform(0, 1, X.shape[0])
        for j in range(X.shape[0]): 
            if p[j] > efficiencies[i]: X[j,broken_wires[i]] = 0
    return X

def downsample(X, size):
    pieces = np.split(X, X.size//size)
    bitarrays = [bitarray(list(p)) for p in pieces]
    X_down = np.asarray([ba2int(b) for b in bitarrays])
    return X_down

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
    print(data.shape)
    input_data = data[:,0:2] # [px, pvx]

    n_active = data[:,2]
    first_active = data[:,3]

    return [input_data, n_active.astype(int), first_active.astype(int)]

def scale(dataset):
    in_data, n_active, first_active = dataset
    
    # scale input data (x, vx)
    weights = 1/np.sqrt(in_data[:,0]**2 + in_data[:,1]**2)
    scaler = StandardScaler()
    scaler.fit(in_data, sample_weight=weights)
    in_data_scaled = scaler.transform(in_data)
    
    enc = OneHotEncoder(sparse=False)
    activations = np.concatenate((n_active.reshape((-1,1)), 
                                  first_active.reshape((-1,1))), axis=1)
    enc.fit(activations)
    activations = enc.transform(activations)
    
    return [in_data_scaled, activations], enc