import numpy as np
import ROOT as r
from os import listdir
from tqdm import tqdm
import pandas as pd

###########################################################################################################
## This script is intended to be run only once.
##  Creates a .csv file from a list of .root files in a given dir (INPUT_DIR)
##  Format of .csv:
##    x1 y1 vx1 vy1 delta(x) delta(y) delta(vx) delta(vy) radius
###########################################################################################################

INPUT_DIR   = '/home/ruben/fewSamples/'
OUTPUT_FILE = '/home/ruben/fewSamples/training_samples.csv' 

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
    "19p8_20": 2
}

def load(in_dir):
    """
	Reads the input file with the muon data and return the dataset as a numpy array
	"""
    thedata = []
    for name in listdir(in_dir):
        for key in radius:
            if key in name:
                r_pipe = radius[key]
        f = r.TFile(input_dir + '/' + name)
        for ev in tqdm(f.globalReco, total=f.globalReco.GetEntries(), desc='Loading data: ' + name):
            if ev.type1 != 3 or ev.type2 != 3:
                continue
            if abs(ev.px1) > 80 or abs(ev.py1) > 80 or abs(ev.pvx1) > 1.5 or abs(ev.pvy1) > 1.5:
                continue
            if abs(ev.px2) > 80 or abs(ev.py2) > 80 or abs(ev.pvx2) > 1.5 or abs(ev.pvy2) > 1.5:
                continue
            thedata.append([ev.px1, ev.py1, ev.pvx1, ev.pvy1,
                            ev.px2 - ev.px1 + 39 * 2 * ev.pvx1, ev.py2 - ev.py1 + 39 * 2 * ev.pvy1,
                            ev.pvx2 - ev.pvx1, ev.pvy2 - ev.pvy1, r_pipe])
    data = np.asarray(thedata)
    print('Data successfully loaded')
    return data

if __name__== "__main__":
    data = load(INPUT_DIR)
    pd.DataFrame(data).to_csv(OUTPUT_FILE, header=False, index=False)
