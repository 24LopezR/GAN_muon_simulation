import numpy as np
import ROOT as r
from os import listdir
from tqdm import tqdm
from array import array
import math

'''
-------------------------------------------------------------------------------
Este script sirve para calcular el Point of Closest Approach a los eventos 
(G4 y GAN) guardar el output en un archivo .root.
Como input toma el .root file creado con generateSamplestoROOT.py
La información que se guarda en el .root es:
- El grosor de la tubería que corresponde al evento               [R]
- Las coordenadas del PoCA                                        [X/Y/Z_PoCA]
- El ángulo de scattering del muon (entre -pi/2 y pi/2)           [theta]
-------------------------------------------------------------------------------
'''

################################ CONSTANTS ####################################
GENERATED_SAMPLES = 'gensamples2'
OUTPUT_FILE = './rootFilesGen/PoCA_'+GENERATED_SAMPLES+'.root'
GENSAMPLES_FILE = './rootFilesGen/'+GENERATED_SAMPLES+'.root'
###############################################################################

ABSPATH = '/'.join(__file__.split('/')[:-2])
print('Project absolute path: ', ABSPATH)

# Dictionary with radius
radius = {
    "18p0_20": 20,
    "18p2_20": 18,
    "18p4_20": 16,
    "18p6_20": 14,
    "19p0_20": 10,
    "19p2_20": 8,
    "19p4_20": 6,
    "19p6_20": 4,
}

def computePoCA(ev, GAN=False):
    p1 = np.asarray([ev.px1, ev.py1, ev.pz1])
    d1 = np.asarray([ev.pvx1, ev.pvy1, ev.pvz1])
    if not GAN:
        p2 = np.asarray([ev.px2, ev.py2, ev.pz2])
        d2 = np.asarray([ev.pvx2, ev.pvy2, ev.pvz2])
    else:
        p2 = np.asarray([ev.px2_gan, ev.py2_gan, ev.pz2_gan])
        d2 = np.asarray([ev.pvx2_gan, ev.pvy2_gan, ev.pvz2_gan])
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
    if theta>math.pi/2: theta -= math.pi
    return poca, theta

if __name__== "__main__":
    file_out = r.TFile.Open(OUTPUT_FILE, "RECREATE")
    R = array('i', [0])
    X_PoCA_G4 = array('f', [0])
    Y_PoCA_G4 = array('f', [0])
    Z_PoCA_G4 = array('f', [0])
    theta_G4  = array('f', [0])
    X_PoCA_GAN = array('f', [0])
    Y_PoCA_GAN = array('f', [0])
    Z_PoCA_GAN = array('f', [0])
    theta_GAN  = array('f', [0])
    tree_out = {}
    for key in radius:
        tree_out[key] = r.TTree(f"tree_{radius[key]}mm", f"tree_{radius[key]}mm")
        tree_out[key].Branch('R', R, 'R/I')
        tree_out[key].Branch('X_PoCA_G4', X_PoCA_G4, 'X_PoCA_G4/F')
        tree_out[key].Branch('Y_PoCA_G4', Y_PoCA_G4, 'Y_PoCA_G4/F')
        tree_out[key].Branch('Z_PoCA_G4', Z_PoCA_G4, 'Z_PoCA_G4/F')
        tree_out[key].Branch('theta_G4', theta_G4, 'theta_G4/F')
        tree_out[key].Branch('X_PoCA_GAN', X_PoCA_GAN, 'X_PoCA_GAN/F')
        tree_out[key].Branch('Y_PoCA_GAN', Y_PoCA_GAN, 'Y_PoCA_GAN/F')
        tree_out[key].Branch('Z_PoCA_GAN', Z_PoCA_GAN, 'Z_PoCA_GAN/F')
        tree_out[key].Branch('theta_GAN', theta_GAN, 'theta_GAN/F')

    f = r.TFile(GENSAMPLES_FILE, "READ")
    for key in radius:
        treename = f'f.tree_{radius[key]}mm'
        print(f'Processing tree {treename}...')
        for ev in tqdm(eval(treename), total=eval(treename).GetEntries()):
            if abs(ev.px1) > 50 or abs(ev.py1) > 50 or abs(ev.pvx1) > 1.5 or abs(ev.pvy1) > 1.5:
                continue
            if abs(ev.px2) > 50 or abs(ev.py2) > 50 or abs(ev.pvx2) > 1.5 or abs(ev.pvy2) > 1.5:
                continue
            R[0] = ev.R
            poca, th = computePoCA(ev, GAN=False)
            X_PoCA_G4[0] = poca[0]
            Y_PoCA_G4[0] = poca[1]
            Z_PoCA_G4[0] = poca[2]
            theta_G4[0] = th
            poca, th = computePoCA(ev, GAN=True)
            X_PoCA_GAN[0] = poca[0]
            Y_PoCA_GAN[0] = poca[1]
            Z_PoCA_GAN[0] = poca[2]
            theta_GAN[0] = th
            tree_out[key].Fill()
    file_out.Write()
    file_out.Close()
