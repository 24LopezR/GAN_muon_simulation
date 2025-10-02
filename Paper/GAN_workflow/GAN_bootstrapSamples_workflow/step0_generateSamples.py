import sys
sys.path.append('/home/ruben/Documents/GAN_muon_simulation/Paper')
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from matplotlib import pyplot
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from optparse import OptionParser
from scipy.stats import skew, kstest, wasserstein_distance
import time
from Evaluation import Evaluation
from joblib import load
from Common.Constants import WEIGHTED_SCALER, EVALUATION_SAMPLES_PATH, MODEL_PATH
import ROOT as r
from array import array
from tqdm import tqdm

'''
-------------------------------------------------------------------------------
Este script sirve para generar eventos con el modelo condicional y guardarlos
en un archivo .root
La información que se guarda en el .root es:
- El grosor de la tubería que corresponde al evento               [R]
- Las variables del muon en el primer detector                    [p*1]
- Las variables reales (G4) del muon en el segundo detector       [p*2]
- Las variables generadas del muon en el segundo detector         [p*2_gan]

Las variables son transformadas de vuelta a las originales simuladas con G4.
-------------------------------------------------------------------------------
'''

############################################################################
### Constants
SCALER = WEIGHTED_SCALER
EVAL_DATA = '/home/ruben/Documents/GAN_muon_simulation/Paper/boostrapWorkflow/DatosBootstrap_forGAN/'
MODEL_FILE = MODEL_PATH
LATENT_DIM = 16
RADIUS = [4,6,8,10,12,14,16,18,20]
radius = {}
for i in [2,4,6,8,10,12,14,16,18,20]: radius[f'_{i}mm'] = i

NSAMPLES = 300000
NIMAGES = 10
OUTPUT_DIR = './DatosGAN/'
############################################################################

def load(rootFile):
    """
	Reads the input file with the muon data and return the dataset as a numpy array
    """
    thedata = []
    for key in radius:
        if key in rootFile:
            r_pipe = radius[key]
    f = r.TFile(rootFile, 'READ')
    for ev in tqdm(f.globalReco, total=f.globalReco.GetEntries(), desc='Loading data: ' + rootFile):
        if ev.type1 != 3 or ev.type2 != 3:
            continue
        if abs(ev.px1) > 80 or abs(ev.py1) > 80 or abs(ev.pvx1) > 1.5 or abs(ev.pvy1) > 1.5:
            continue
        if abs(ev.px2) > 80 or abs(ev.py2) > 80 or abs(ev.pvx2) > 1.5 or abs(ev.pvy2) > 1.5:
            continue
        thedata.append([ev.px1, ev.py1, ev.pvx1, ev.pvy1,
                        ev.px2 - ev.px1 + 39 * 2 * ev.pvx1, ev.py2 - ev.py1 + 39 * 2 * ev.pvy1,
                        ev.pvx2 - ev.pvx1, ev.pvy2 - ev.pvy1])
    data = np.asarray(thedata)
    print('Data successfully loaded')
    return data, r_pipe

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # GPU memory usage configuration
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

    print('Loading model...')
    g_model = keras.models.load_model(MODEL_FILE)

    encoder = OneHotEncoder(categories=[[4,6,8,10,14,16,18,20]], sparse_output=False)
    for f in os.listdir(EVAL_DATA)i[0:10]:
        if '_2mm' in f: continue
        if not any(s in f for s in [f'_{i}.root' for i in range(1000,1010)]): 
           print(f'    Skipping {f}')
           continue
        print(f'Loading_data from {f}...')
        variables, r_pipe = load(f'{EVAL_DATA}/{f}')
        variables = SCALER.transform(variables)
        if '_12mm_' in f:
            labels = np.zeros((variables.shape[0], 8))
            labels[:,3:5] = 0.5
        else:
            labels = r_pipe * np.ones(variables.shape[0])
            labels = encoder.fit_transform(labels.reshape(-1,1))
        #print(variables.shape, labels.shape)
        #print(variables[0], labels[0])
        dataset = np.hstack([variables, labels])
        print('Number of evaluation samples = '+str(dataset.shape[0]))
    
        print('Generating evaluation samples...')
        start_time = time.time()
        z_noise = np.random.normal(size=(dataset.shape[0], LATENT_DIM))
        fake_samples = g_model.predict([z_noise, np.hstack([dataset[:,0:4], dataset[:,8:]])],
                                       batch_size=dataset.shape[0])
        end_time = time.time()
        print('    Start time = '+str(start_time))
        print('    End time = '+str(end_time))
        print("--- Generation time: %s seconds ---" % (end_time - start_time))
    
        g4_dataset = SCALER.inverse_transform(dataset[:,0:8])
        gan_dataset = SCALER.inverse_transform(np.hstack([dataset[:,0:4], fake_samples]))
        '''print(f'Consistency ckeck')
        print(f' G4 dataset shape:  {np.shape(g4_dataset)}')
        print(f'                    {g4_dataset[0:2]}')
        print(f' GAN dataset shape: {np.shape(gan_dataset)}')
        print(f'                    {gan_dataset[0:2]}')'''
        '''
        # Fill ROOT file with generated samples
        OUTPUT_FILE = f'{OUTPUT_DIR}/GAN_{f}'
        print(f'Writing to {OUTPUT_FILE}')
        f_out = r.TFile.Open(OUTPUT_FILE, "RECREATE")
        # COntainers for variables
        R = array('i', [0])
        px1 = array('f', [0])
        py1 = array('f', [0])
        pz1 = array('f', [0])
        pvx1 = array('f', [0])
        pvy1 = array('f', [0])
        pvz1 = array('f', [0])
        px2 = array('f', [0])
        py2 = array('f', [0])
        pz2 = array('f', [0])
        pvx2 = array('f', [0])
        pvy2 = array('f', [0])
        pvz2 = array('f', [0])
        px2_gan = array('f', [0])
        py2_gan = array('f', [0])
        pz2_gan = array('f', [0])
        pvx2_gan = array('f', [0])
        pvy2_gan = array('f', [0])
        pvz2_gan = array('f', [0])
        # build one tree per radius value
        tree = r.TTree(f"globalReco", f"globalReco")
        tree.Branch('R', R, 'R/I')
        tree.Branch('px1', px1, 'px1/F')
        tree.Branch('py1', py1, 'py1/F')
        tree.Branch('pz1', pz1, 'pz1/F')
        tree.Branch('pvx1', pvx1, 'pvx1/F')
        tree.Branch('pvy1', pvy1, 'pvy1/F')
        tree.Branch('pvz1', pvz1, 'pvz1/F')
        tree.Branch('px2', px2, 'px2/F')
        tree.Branch('py2', py2, 'py2/F')
        tree.Branch('pz2', pz2, 'pz2/F')
        tree.Branch('pvx2', pvx2, 'pvx2/F')
        tree.Branch('pvy2', pvy2, 'pvy2/F')
        tree.Branch('pvz2', pvz2, 'pvz2/F')
        tree.Branch('px2_gan', px2_gan, 'px2_gan/F')
        tree.Branch('py2_gan', py2_gan, 'py2_gan/F')
        tree.Branch('pz2_gan', pz2_gan, 'pz2_gan/F')
        tree.Branch('pvx2_gan', pvx2_gan, 'pvx2_gan/F')
        tree.Branch('pvy2_gan', pvy2_gan, 'pvy2_gan/F')
        tree.Branch('pvz2_gan', pvz2_gan, 'pvz2_gan/F')
        for N in tqdm(range(g4_dataset.shape[0])):
            R[0] = int(r_pipe)
            px1[0] = g4_dataset[N,0]
            py1[0] = g4_dataset[N,1]
            pz1[0] = 39.
            pvx1[0] = g4_dataset[N,2]
            pvy1[0] = g4_dataset[N,3]
            pvz1[0] = 1.
            px2[0] = g4_dataset[N,4] + px1[0] - 2*39*pvx1[0]
            py2[0] = g4_dataset[N,5] + py1[0] - 2*39*pvy1[0]
            pz2[0] = -39.
            pvx2[0] = g4_dataset[N,6] + pvx1[0]
            pvy2[0] = g4_dataset[N,7] + pvy1[0]
            pvz2[0] = 1.
            px2_gan[0] = gan_dataset[N,4] + px1[0] - 2*39*pvx1[0]
            py2_gan[0] = gan_dataset[N,5] + py1[0] - 2*39*pvy1[0]
            pz2_gan[0] = -39.
            pvx2_gan[0] = gan_dataset[N,6] + pvx1[0]
            pvy2_gan[0] = gan_dataset[N,7] + pvy1[0]
            pvz2_gan[0] = 1.
            tree.Fill()
        f_out.Write()
        f_out.Close()
        '''
    print('Generation done!')
