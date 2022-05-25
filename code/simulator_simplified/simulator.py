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

def shortest_distance(wire_x, wire_z, x_0, z_0, theta):
    return np.abs(np.cos(theta)*(wire_x-x_0) - np.sin(theta)*(wire_z-z_0))

def compute_activations(wire_pos, wire_z, x_0, z_0, theta):
    d = np.asarray([shortest_distance(w, wire_z, x_0, z_0, theta) for w in wire_pos])
    return (d < epsilon)*1

if __name__ == "__main__":
    
    # Constants
    n_events = 400000
    nw = 216
    spacing = 0.4 # cm
    epsilon = 0.05 # cm
    wire_pos = [(w-nw//2)*spacing for w in range(nw)]
    z_0 = 1
    wire_z = 0
    
    # Initialize data matrix
    in_data     = []
    activations = []
    
    print('Progress:')
    while len(in_data) <= (n_events):
        # Generate random p variables
        theta_temp = np.pi/2 * np.random.uniform(low=-1, high=1)
        x_temp = 40 * np.random.uniform(low=-1, high=1)

        # Compute activated wires
        act = compute_activations(wire_pos=wire_pos, wire_z=wire_z,
                                  x_0=x_temp, z_0=z_0, theta=theta_temp)

        if np.sum(act) > 0 and np.sum(act) <= 5:
            in_data.append([x_temp, np.tan(theta_temp)])
            to_save_data = [np.sum(act), act.nonzero()[0][0]]
            activations.append(to_save_data)
            if len(in_data) % 5 == 0:
                print("\r", end="")
                print('Events generated: '+str(len(in_data)), end='')
    
    in_data     = np.asarray(in_data) # [px, pvx]
    activations = np.asarray(activations)
    data = np.hstack((in_data, activations))

    plt.hist(in_data[:,0], range=(-40,40), bins=200)
    plt.hist(in_data[:,1], range=(-10,10), bins=200)
    plt.show()
    pd.DataFrame(data).to_csv("/home/ruben/GAN_muon_simulation/data/sim2.csv", header=False, index=False)