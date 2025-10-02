import numpy as np
import ROOT as r
from os import listdir
from tqdm import tqdm
from array import array

ABSPATH = '/'.join(__file__.split('/')[:-2])
print('Project absolute path: ', ABSPATH)

POCA_FILE = './rootFiles/PoCA_evaluationSamples_G4.root'

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

if __name__== "__main__":
    """
	Reads the input file with the muon data and return the dataset as a numpy array
	"""
    # Create hists
    hists = {}
    for key in radius:
        hists[radius[key]] = [r.TH2F(f"g4_{radius[key]}_N", f"[G4] Number of points YZ ({radius[key]} mm); Y; Z", 100, -50, 50, 100, -39, 39),
                              r.TH2F(f"g4_{radius[key]}_theta", f"[G4] scattering angle YZ ({radius[key]} mm); Y; Z", 100, -50, 50, 100, -39, 39),
                              r.TH2F(f"g4_{radius[key]}_theta2", f"[G4] scattering angle squared YZ ({radius[key]} mm); Y; Z", 100, -50, 50, 100, -39, 39)]

    file = r.TFile.Open(POCA_FILE, "READ")
    fileout = r.TFile.Open("./rootFiles/PoCAmaps_evaluationSamples_G4.root", "RECREATE")
    tree = file.Get("tree")
    for ev in tqdm(tree, total=tree.GetEntries()):
        hists[ev.R][0].Fill(ev.Y_PoCA, ev.Z_PoCA)
        hists[ev.R][1].Fill(ev.Y_PoCA, ev.Z_PoCA, ev.theta)
        hists[ev.R][2].Fill(ev.Y_PoCA, ev.Z_PoCA, ev.theta*ev.theta)
    file.Close()

    for key in hists: 
        hists[key][0].Write()
        hists[key][1].Write()
        hists[key][2].Write()
        h_thetasqmean = hists[key][2].Clone()
        h_thetasqmean.Divide(hists[key][0])
        h_thetameansq = hists[key][1].Clone()
        h_thetameansq.Divide(hists[key][0])
        h_thetameansq.Multiply(h_thetameansq)
        h_thetameansq.Scale(-1)
        h_RMS_G4 = r.TH2F(f"g4_{key}_RMS", f"[G4] scattering angle RMS YZ ({key} mm); Y; Z", 100, -50, 50, 100, -39, 39)
        h_RMS_G4.Add(h_thetasqmean)
        h_RMS_G4.Add(h_thetameansq)
        h_RMS_G4.Write()
    fileout.Close()
