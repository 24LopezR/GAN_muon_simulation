# Generative Adversarial Networks for Simulation in Muon Tomography

Repository for code, data and results. It contains 3 subdirectories:

## Paper
Code ready to evaluate the conditional model.
Stuff related to the paper: code, datafiles and trained model.

### Requirements
A list of the packages needed to run (with the version I use):
- python (3.9.16) any python 3.X will work
- tensorflow (2.10.0)
- keras (2.10.0)
- numpy (1.26.4)
- scikit-learn (1.4.0)
- scipy (1.12.0)
- pandas (2.2.0)
- joblib (1.3.2)
- matplotlib (3.8.2)

#### How I setup my conda environment
```bash
conda create --name=tf-root python=3.9
conda install -c conda-forge root
conda install -c conda-forge cudatoolkit=11.2.2 cudnn=8.1.0
python3 -m pip install tensorflow==2.10
pip install pandas scikit-learn scipy matplotlib
```

#### Specify the location of evaluation samples
In ```Paper/Common/Constants.py```it is necessary to specify the location of the .csv file with the samples used for evaluation.
Currently, this file has a size of 493 MB.

### Instructions to run

#### Generated ROOT files and Point of Closest Approach (PoCA) maps:
The generation of the PoCA maps for G4 samples and GAN samples is done via 3 steps:
1. ```generateSamplestoROOT.py``` uses the GAN model to generate muon events and save them in a .root file.
The info saved in the .root is:
    - The thickness of the pipe corresponding to the event            (R)
    - The first detector variables                                    (p*1)
    - The second detector variables (G4)                              (p*2)
    - The second detector variables generated with GAN model          (p*2_gan)

3. ```runPoCAfromROOT_GANsamples.py```uses the output of previous step to run the PoCA algorithm and save the info to a .root file.
4. ```plotPoCAmaps_GAN.py``` generates 2D hists os PoCA maps for G4 samples and GAN samples.

I already prerun the 3 steps, so to test you can skip any of them. The generated root files are in https://cernbox.cern.ch/s/l8RVUAVHXACFtmU (rootFilesGen/).


#### Old evaluation via Evaluatin.py
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
