# Generative Adversarial Networks for Simulation in Muon Tomography

Repository for code, data and results. It contains 3 subdirectories:

## Paper
Code ready to evaluate the conditional model.
Stuff related to the paper: code, datafiles and trained model.

### Requirements
A list of the packages needed to run (with the version I use):
- python (3.9.16) any python 3.X will work
- tensorflow-gpu (2.4.1)
- keras (2.4.3)
- numpy (1.23.5)
- scikit-learn (1.2.2)
- pandas (1.5.3)
- joblib (1.1.1)
- matplotlib (3.7.1)
#### Specify the location of evaluation samples
In ```Paper/Common/Constants.py```it is necessary to specify the location of the .csv file with the samples used for evaluation.
Currently, this file has a size of 493 MB.

### Instructions to run
The script ```plotEvaluation.py```contains code to:
1. load the .h5 model placed in ```Paper/Common/Models/v1/muon_propagation_WGAN_model.h5```
2. load the evaluation samples located in a .csv file.
3. evaluate numerically the model (mean, covariance matrix...)
4. produce plots of real and generated data to a .pdf file.

To evaluate the model, just run the following command:
```
python3 plotEvaluation.py
```

## Pytorch [In Development]
Pytorch implementation of WGAN-GP (from https://github.com/EmilienDupont/wgan-gp).

## Training 
Old code I developed for TFM.
