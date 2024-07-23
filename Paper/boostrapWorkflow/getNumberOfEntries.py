import os
import numpy as np
import ROOT as r
import json

# Dictionary with radius
radius = {
    "18p0_20": 20,
    "18p2_20": 18,
    "18p4_20": 16,
    "18p6_20": 14,
    "18p8_20": 12,
    "19p0_20": 10,
    "19p2_20": 8,
    "19p4_20": 6,
    "19p6_20": 4,
    "19p8_20": 2,
}

DATA_DIR = '/home/ruben/Documents/DatosPipes/'

if __name__=='__main__':

    master_dict = dict()

    for f in os.listdir(DATA_DIR):
        #print(f)
        # Get radius of file
        th = 0
        for key in radius:
            if key in f: th = radius[key]
        if th == 0:
            print('Unknown thickness for file'); exit()
        if str(th) not in master_dict: master_dict[str(th)] = dict()

        _file = r.TFile.Open(f'{DATA_DIR}/{f}', 'r')
        _tree = _file.Get("globalReco")
        #print(_tree.GetEntries())
        master_dict[str(th)][f] = _tree.GetEntries()

    print(json.dumps(master_dict, sort_keys=True, indent=4, separators=(',', ': ')))
    #print(json.dumps(master_dict))
