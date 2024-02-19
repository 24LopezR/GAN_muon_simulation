import numpy as np
import ROOT as r
import os
from os import listdir
from tqdm import tqdm
from array import array
import math
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-i', '--infile', dest='infilename')
args = parser.parse_args()
infilename = args.infilename

ABSPATH = '/'.join(__file__.split('/')[:-2])
print('Project absolute path: ', ABSPATH)

DATA_SAMPLES_PATH = '/home/ruben/Documents/DatosPipes/'
OUTPUT_PATH       = '/home/ruben/Documents/DatosPipesPoCA/'

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

def computePoCA(ev):
    p1 = np.asarray([ev.px1, ev.py1, ev.pz1])
    d1 = np.asarray([ev.pvx1, ev.pvy1, ev.pvz1])
    p2 = np.asarray([ev.px2, ev.py2, ev.pz2])
    d2 = np.asarray([ev.pvx2, ev.pvy2, ev.pvz2])
    # check that both trajectories are not parallel
    n = np.cross(d1,d2)
    if np.linalg.norm(n) < 1e-9:
        print('Trajectories are parallel')
        return
    # first compute the points of each trajectory that are closer to the other trajectory
    A = np.dot(p2-p1,d1)
    B = np.dot(d2,d1)
    C = np.dot(d1,d1)
    D = np.dot(p2-p1,d2)
    E = np.dot(d2,d2)
    t1 = (A*E-B*D)/(C*E-B*B)
    t2 = -(B*A-C*D)/(B*B-C*E)
    P1 = p1 + t1*d1
    P2 = p2 + t2*d2
    # poca will be the middle point
    poca = (P1+P2)/2
    # compute also the scattering angle
    costheta = np.dot(d1,d2)/(np.linalg.norm(d1)*np.linalg.norm(d2))
    theta = np.arccos(costheta)
    #if theta>math.pi/2: theta -= math.pi
    return poca, theta

if __name__== "__main__":
    if not os.path.exists(OUTPUT_PATH): os.makedirs(OUTPUT_PATH)
    print(f'Input file: {infilename}')

    # Out file definition
    #outfilename = f'{OUTPUT_PATH}/PoCA_{infilename}'
    outfilename = f'./PoCA_test_20.root'
    print(f'Output file: {outfilename}')
    outfile = r.TFile.Open(outfilename, "RECREATE")
    outree = r.TTree("tree", "tree")
    R = array('i', [0])
    X_PoCA = array('f', [0])
    Y_PoCA = array('f', [0])
    Z_PoCA = array('f', [0])
    theta  = array('f', [0])
    outree.Branch('R', R, 'R/I')
    outree.Branch('X_PoCA', X_PoCA, 'X_PoCA/F')
    outree.Branch('Y_PoCA', Y_PoCA, 'Y_PoCA/F')
    outree.Branch('Z_PoCA', Z_PoCA, 'Z_PoCA/F')
    outree.Branch('theta', theta, 'theta/F')
    
    # In file definition
    for key in radius:
        if key in infilename:
            r_pipe = radius[key]
    print(f'R = {str(r_pipe)} mm')
    f = r.TFile(f'{DATA_SAMPLES_PATH}/{infilename}', 'READ')
    for i,ev in enumerate(f.globalReco):#, total=f.globalReco.GetEntries(), desc='Loading data: ' + name):
        if i%5000==0: print(f'Event {i}...')
        if ev.type1 != 3 or ev.type2 != 3:
            continue
        if abs(ev.px1) > 80 or abs(ev.py1) > 80 or abs(ev.pvx1) > 1.5 or abs(ev.pvy1) > 1.5:
            continue
        if abs(ev.px2) > 80 or abs(ev.py2) > 80 or abs(ev.pvx2) > 1.5 or abs(ev.pvy2) > 1.5:
            continue
        poca, th = computePoCA(ev)
        R[0] = r_pipe
        X_PoCA[0] = poca[0]
        Y_PoCA[0] = poca[1]
        Z_PoCA[0] = poca[2]
        theta[0] = th
        outree.Fill()
    outfile.Write()
    outfile.Close()
    f.Close()
