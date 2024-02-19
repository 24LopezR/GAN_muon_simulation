import numpy as np
import ROOT as r
from os import listdir
from tqdm import tqdm
from array import array
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-i', '--infile', dest='infilename')
args = parser.parse_args()
infilename = args.infilename

'''
-------------------------------------------------------------------------------
Este script sirve para producir histogramas 2D de los mapas de los PoCA
Toma como input el archivo .root creado con step1_runPoCAonG4.py.
-------------------------------------------------------------------------------
'''

################################ CONSTANTS ####################################
POCA_FILE_PATH = '/home/ruben/Documents/PoCA_G4/'
OUTPUT_FILE_PATH = "/home/ruben/Documents/PoCAmaps_50bins/"
MAX_HIST = 1.
MIN_HIST = 1e-4
NBINSX = 50
NBINSY = 50
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


def saveSingleHist(infilename):
    # Create hists
    hists = [r.TH2F(f"h2_N",      f"Number of points YZ; Y; Z", NBINSX, -50, 50, NBINSY, -50, 50),
             r.TH2F(f"h2_theta",  f"Scattering angle YZ; Y; Z", NBINSX, -50, 50, NBINSY, -50, 50),
             r.TH2F(f"h2_theta2", f"Scattering angle squared YZ; Y; Z", NBINSX, -50, 50, NBINSY, -50, 50)]

    f = r.TFile.Open(f'{POCA_FILE_PATH}/{infilename}', "READ")
    tree = f.Get('tree')
    for key in radius:
        for ev in tree:
            hists[0].Fill(ev.Y_PoCA, ev.Z_PoCA)
            hists[1].Fill(ev.Y_PoCA, ev.Z_PoCA, ev.theta)
            hists[2].Fill(ev.Y_PoCA, ev.Z_PoCA, ev.theta*ev.theta)
    f.Close()

    outputfilename = f'{OUTPUT_FILE_PATH}/map_{infilename}'
    fileout = r.TFile.Open(outputfilename, "RECREATE")
    hists_RMS = {}
    hists[0].Write()
    hists[1].Write()
    hists[2].Write()
    h_thetasqmean = hists[2].Clone()
    h_thetasqmean.Divide(hists[0])
    h_thetameansq = hists[1].Clone()
    h_thetameansq.Divide(hists[0])
    h_thetameansq.Multiply(h_thetameansq)
    h_thetameansq.Scale(-1)
    h_RMS = r.TH2F(f"h2_RMS", f"Scattering angle RMS YZ; Y; Z", NBINSX, -50, 50, NBINSY, -50, 50)
    h_RMS.Add(h_thetasqmean)
    h_RMS.Add(h_thetameansq)
    h_RMS.SetMaximum(MAX_HIST)
    h_RMS.SetMinimum(MIN_HIST)
    h_RMS.Write()
    h_RMS_a = r.TH2F(f"h2_RMS_a", f"Scattering angle RMS (approx) YZ; Y; Z", NBINSX, -50, 50, NBINSY, -50, 50)
    h_RMS_a = h_thetasqmean
    h_RMS_a.Write()
    fileout.Write()
    fileout.Close()

if __name__== "__main__":
    if not os.path.exists(OUTPUT_FILE_PATH): os.makedirs(OUTPUT_FILE_PATH)
    for infilename in tqdm(listdir(POCA_FILE_PATH)):
        if not '.root' in infilename: continue
        saveSingleHist(infilename)
