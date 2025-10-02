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
#from argparse import ArgumentParser
''' DISABLED BECAUSE argparse WAS NOT WORKING IN MY ENVIRONMENT
parser = argparse.ArgumentParser(
                    prog='plotEvaluation',
                    description='Evaluate the performance of a model')
parser.add_argument('-m', '--model', default=MODEL_PATH, help='Path to the model to evaluate. Must be a .h5 file.')
parser.add_argument('-o', '--output', default='./evaluationPaper.pdf', help='Name of the output pdf file.')
args = parser.parse_args()
'''

############################################################################
### Constants
SCALER = WEIGHTED_SCALER
EVAL_DATA_FILE = EVALUATION_SAMPLES_PATH + "/evaluationSamples_seed19.csv"
MODEL_FILE = MODEL_PATH
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
    variables = SCALER.transform(variables)

    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(labels.reshape((-1, 1)))
    labels = encoder.transform(labels.reshape((-1, 1)))
    return np.concatenate([variables, labels], axis=1), SCALER

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # GPU memory usage configuration
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

    # Read user options
    LATENT_DIM = 16

    print('Loading model...')
    g_model = keras.models.load_model(MODEL_FILE)

    print('Loading_data...')
    dataset, SCALER = load(EVAL_DATA_FILE)
    n = 300000
    data_reduced = np.zeros((8*n,dataset.shape[1]))
    for i in range(8):
        data_reduced[n*i:n*(i+1)] = dataset[dataset[:,8:].argmax(1)==i][0:n]
    dataset = data_reduced
    print('Number of evaluation samples = '+str(dataset.shape[0]))

    interpolation_data = pd.read_csv(EVAL_DATA_FILE).to_numpy()
    interpolation_data = interpolation_data[interpolation_data[:,8]==12][0:n,0:8]
    interpolation_data = SCALER.transform(interpolation_data)
    interpolation_labels = np.zeros((n,8))
    interpolation_labels[:,3:5] = 0.5

    eval_interpolation = Evaluation(g_model, LATENT_DIM, dataset=np.hstack([interpolation_data, interpolation_labels]),
                                        scaler=SCALER)
    eval_ = Evaluation(g_model, LATENT_DIM, dataset=dataset, scaler=SCALER)

    print('Generating evaluation samples...')
    #start_time = time.time()
    #eval_interpolation.generate_evaluation_samples()
    #eval_.generate_evaluation_samples()
    #end_time = time.time()
    #print('    Start time = '+str(start_time))
    #print('    End time = '+str(end_time))
    #print("--- Generation time: %s seconds ---" % (end_time - start_time))
    real_dataset = eval_.real_samples
    #fake_dataset = eval_.fake_samples
    #labels = eval_.label_radius
    #real_inter = eval_interpolation.real_samples
    #fake_inter = eval_interpolation.fake_samples

    # Save eval samples
    print('Saving evaluation samples...')
    np.save('baseEvalRealData_seed19', real_dataset)
    #np.save('baseEvalGenData', fake_dataset)
    #np.save('baseEvalRealInterpol', real_inter)
    #np.save('baseEvalGenInterpol', fake_inter)
    #np.save('baseEvalLabels', labels)
