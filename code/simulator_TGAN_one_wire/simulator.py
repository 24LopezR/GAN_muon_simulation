#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 13:14:53 2022

File to generate data that simulates wire activation.

@author: ruben
"""
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import pandas as pd
import ROOT

def shortest_distance(P, Q, v):
    vnorm = norm(v, axis=0)
    if vnorm.any() == 0:
        return 0
    else:
        QP = Q-P
        cross = np.cross(QP, v, axis=0)
        d = norm(cross, axis=0)/vnorm
        #print(np.min(d))
        return d

def compute_activations(P, Q, v):
    d = shortest_distance(P, Q, v)
    return (d < epsilon)
    

def sample_values(px_sample, py_sample, pvx_sample, pvy_sample):
    if px_sample.size == pvx_sample.size:
        i = np.random.randint(0, px_sample.size)
        return px_sample[i], py_sample[i], pvx_sample[i], pvy_sample[i]

if __name__ == "__main__":
    
    # Load data
    f = ROOT.TFile('/home/ruben/Documents/TFM/GAN_muon_simulation/data/PipeTest_18p4_20_fullformat_18p4_20_seed3.root', 'READ')
    dataset = ROOT.RDataFrame('globalReco', f).AsNumpy()
    px  = dataset['px1']
    pvx = dataset['pvx1']
    py  = dataset['py1']
    pvy = dataset['pvy1']
    samples = len(px)
    
    # Constants
    n_events = 100
    nw = 216
    spacing = 0.4
    wirex, wirey = np.meshgrid(spacing*np.arange(-nw//2,nw//2),spacing*np.arange(-nw//2,nw//2))
    P = np.array([wirex, wirey, np.zeros((nw,nw))])
    epsilon = 0.02
    z0 = 1
    
    # Initialize data matrix
    in_data     = []
    activations = []
    
    print('Progress:')
    while len(in_data) <= (n_events):
    #for s in range(samples):
        # Generate random p variables
        px_temp, py_temp, pvx_temp, pvy_temp = sample_values(px, py, pvx, pvy)
        Q = np.array([px_temp*np.ones((nw,nw)), py_temp*np.ones((nw,nw)), np.zeros((nw,nw))])
        v = np.array([pvx_temp*np.ones((nw,nw)), pvy_temp*np.ones((nw,nw)), -np.ones((nw,nw))])
        in_data_temp = [px_temp, py_temp, pvx_temp, pvy_temp]
        # Compute activated wires
        act = compute_activations(P, Q, v)
        if np.sum(act) > 0:
            in_data.append(in_data_temp)
            activations.append(act)
            if len(in_data) % 5 == 0:
                print("\r", end="")
                print('Events generated: '+str(len(in_data)), end='')
    
    in_data     = np.asarray(in_data)
    activations = np.asarray(activations)
    hits = np.sum(activations, axis=(1,2))
    plt.hist(hits, range=(0,5), bins=5)
    #pd.DataFrame(data).to_csv("file.csv", header=False, index=False)