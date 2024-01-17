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
EVAL_DATA_FILE = EVALUATION_SAMPLES_PATH + "/evaluationSamples_Oct16.csv"
MODEL_FILE = MODEL_PATH
LATENT_DIM = 16
RADIUS = [4,6,8,10,14,16,18,20]

OUTPUT_FILE = '.rootFilesGen/gensamples2.root'
############################################################################

"""
Loads the evaluation samples into a numpy array
"""
def load(inputfile):
    data = pd.read_csv(inputfile).to_numpy()
    mask = [i in [4, 6, 8, 10, 14, 16, 18, 20] for i in data[:, 8]]
    data = data[mask]
    variables = data[:, :8]
    labels = data[:, 8]
    print(f'Scaler transform: {np.shape(variables)}')
    variables = SCALER.transform(variables)

    encoder = OneHotEncoder(sparse=False)
    encoder.fit(labels.reshape((-1, 1)))
    labels = encoder.transform(labels.reshape((-1, 1)))
    return np.concatenate([variables, labels], axis=1)

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # GPU memory usage configuration
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

    print('Loading model...')
    g_model = keras.models.load_model(MODEL_FILE)

    print('Loading_data...')
    dataset = load(EVAL_DATA_FILE)
    print('Number of evaluation samples = '+str(dataset.shape[0]))

    print('Generating evaluation samples...')
    start_time = time.time()
    z_noise = np.random.normal(size=(dataset.shape[0], LATENT_DIM))
    fake_samples = g_model.predict([z_noise, np.hstack([dataset[:,0:4], dataset[:,8:]])])
    end_time = time.time()
    print('    Start time = '+str(start_time))
    print('    End time = '+str(end_time))
    print("--- Generation time: %s seconds ---" % (end_time - start_time))

    g4_dataset = SCALER.inverse_transform(dataset[:,0:8])
    gan_dataset = SCALER.inverse_transform(np.hstack([dataset[:,0:4], fake_samples]))
    print(f'Consistency ckeck')
    print(f' G4 dataset shape:  {np.shape(g4_dataset)}')
    print(f'                    {g4_dataset[0:2]}')
    print(f' GAN dataset shape: {np.shape(gan_dataset)}')
    print(f'                    {gan_dataset[0:2]}')

    # Fill ROOT file with generated samples
    print('Data:   '+str(dataset[0]))
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
    trees = {}
    for radius in RADIUS: 
        trees[radius] = r.TTree(f"tree_{radius}mm", f"tree_{radius}mm")
        trees[radius].Branch('R', R, 'R/I')
        trees[radius].Branch('px1', px1, 'px1/F')
        trees[radius].Branch('py1', py1, 'py1/F')
        trees[radius].Branch('pz1', pz1, 'pz1/F')
        trees[radius].Branch('pvx1', pvx1, 'pvx1/F')
        trees[radius].Branch('pvy1', pvy1, 'pvy1/F')
        trees[radius].Branch('pvz1', pvz1, 'pvz1/F')
        trees[radius].Branch('px2', px2, 'px2/F')
        trees[radius].Branch('py2', py2, 'py2/F')
        trees[radius].Branch('pz2', pz2, 'pz2/F')
        trees[radius].Branch('pvx2', pvx2, 'pvx2/F')
        trees[radius].Branch('pvy2', pvy2, 'pvy2/F')
        trees[radius].Branch('pvz2', pvz2, 'pvz2/F')
        trees[radius].Branch('px2_gan', px2_gan, 'px2_gan/F')
        trees[radius].Branch('py2_gan', py2_gan, 'py2_gan/F')
        trees[radius].Branch('pz2_gan', pz2_gan, 'pz2_gan/F')
        trees[radius].Branch('pvx2_gan', pvx2_gan, 'pvx2_gan/F')
        trees[radius].Branch('pvy2_gan', pvy2_gan, 'pvy2_gan/F')
        trees[radius].Branch('pvz2_gan', pvz2_gan, 'pvz2_gan/F')
    for N in tqdm(range(g4_dataset.shape[0])):
        R[0] = int(np.dot(dataset[N,8:],RADIUS))
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
        trees[R[0]].Fill()
    f_out.Write()
    f_out.Close()

    print('Generation done!')
