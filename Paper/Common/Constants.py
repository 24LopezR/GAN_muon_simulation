from sklearn.preprocessing import StandardScaler
from joblib import load

ABSPATH = '/'.join(__file__.split('/')[:-2])
print(f'--- ABSPATH: {ABSPATH} ---') 

WEIGHTED_SCALER = load(ABSPATH + '/Common/weighted_scaler.joblib')

TRAINING_SAMPLES_PATH   = '/home/ruben/Documents/trainingSamples'
EVALUATION_SAMPLES_PATH = '/home/ruben/Documents/evaluationSamples'

MODEL_PATH = ABSPATH + '/Common/Models/v1/muon_propagation_WGAN_model.h5'

if __name__=='__main__':
    print('ABSPATH:  ', ABSPATH)
    print(WEIGHTED_SCALER)
    print('Mean:   '+str(WEIGHTED_SCALER.mean_))
    print('Var:    '+str(WEIGHTED_SCALER.var_))
