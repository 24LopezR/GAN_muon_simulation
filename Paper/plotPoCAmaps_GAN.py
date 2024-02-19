import numpy as np
import ROOT as r
from os import listdir
from tqdm import tqdm
from array import array

'''
-------------------------------------------------------------------------------
Este script sirve para producir histogramas 2D de los mapas de los PoCA
Toma como input el archivo .root creado con runPoCAfromROOT_GANsamples.py.
-------------------------------------------------------------------------------
'''

################################ CONSTANTS ####################################
POCA_FILE = './rootFilesGen/PoCA_gensamples2plus12mm.root'
OUTPUT_FILE = "./rootFilesGen/PoCAmaps_gensamples2plus12mm.root"
NBINS = 50
###############################################################################

ABSPATH = '/'.join(__file__.split('/')[:-2])
print('Project absolute path: ', ABSPATH)

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
}

if __name__== "__main__":
    MAX = 1.#np.max([np.max([hists_RMS[key][0].GetMaximum(), hists_RMS[key][1].GetMaximum()]) for key in hists_RMS])
    MIN = 1e-4#np.min([np.min([hists_RMS[key][0].GetMinimum(), hists_RMS[key][1].GetMinimum()]) for key in hists_RMS])
    # Create hists
    hists = {}
    for key in radius:
        hists[radius[key]] = [r.TH2F(f"g4_{radius[key]}_N", f"[G4] Number of points YZ ({radius[key]} mm); Y; Z", NBINS, -50, 50, NBINS, -50, 50),
                              r.TH2F(f"g4_{radius[key]}_theta", f"[G4] scattering angle YZ ({radius[key]} mm); Y; Z", NBINS, -50, 50, NBINS, -50, 50),
                              r.TH2F(f"g4_{radius[key]}_theta2", f"[G4] scattering angle squared YZ ({radius[key]} mm); Y; Z", NBINS, -50, 50, NBINS, -50, 50),
                              r.TH2F(f"gan_{radius[key]}_N", f"[GAN] Number of points YZ ({radius[key]} mm); Y; Z", NBINS, -50, 50, NBINS, -50, 50),
                              r.TH2F(f"gan_{radius[key]}_theta", f"[GAN] scattering angle YZ ({radius[key]} mm); Y; Z", NBINS, -50, 50, NBINS, -50, 50),
                              r.TH2F(f"gan_{radius[key]}_theta2", f"[GAN] scattering angle squared YZ ({radius[key]} mm); Y; Z", NBINS, -50, 50, NBINS, -50, 50)]

    f = r.TFile.Open(POCA_FILE, "READ")
    for key in radius:
        treename = f'f.tree_{radius[key]}mm'
        print(f'Processing {treename}')
        for ev in tqdm(eval(treename), total=eval(treename).GetEntries()):
            hists[ev.R][0].Fill(ev.Y_PoCA_G4, ev.Z_PoCA_G4)
            hists[ev.R][1].Fill(ev.Y_PoCA_G4, ev.Z_PoCA_G4, ev.theta_G4)
            hists[ev.R][2].Fill(ev.Y_PoCA_G4, ev.Z_PoCA_G4, ev.theta_G4*ev.theta_G4)
            hists[ev.R][3].Fill(ev.Y_PoCA_GAN, ev.Z_PoCA_GAN)
            hists[ev.R][4].Fill(ev.Y_PoCA_GAN, ev.Z_PoCA_GAN, ev.theta_GAN)
            hists[ev.R][5].Fill(ev.Y_PoCA_GAN, ev.Z_PoCA_GAN, ev.theta_GAN*ev.theta_GAN)
    f.Close()

    fileout = r.TFile.Open(OUTPUT_FILE, "RECREATE")
    hists_RMS = {}
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
        h_RMS_G4 = r.TH2F(f"g4_{key}_RMS", f"[G4] scattering angle RMS YZ ({key} mm); Y; Z", NBINS, -50, 50, NBINS, -50, 50)
        h_RMS_G4.Add(h_thetasqmean)
        h_RMS_G4.Add(h_thetameansq)
        hists[key][3].Write()
        hists[key][4].Write()
        hists[key][5].Write()
        h_thetasqmean = hists[key][5].Clone()
        h_thetasqmean.Divide(hists[key][3])
        h_thetameansq = hists[key][4].Clone()
        h_thetameansq.Divide(hists[key][3])
        h_thetameansq.Multiply(h_thetameansq)
        h_thetameansq.Scale(-1)
        h_RMS_GAN = r.TH2F(f"gan_{key}_RMS_GAN", f"[GAN] scattering angle RMS YZ ({key} mm); Y; Z", NBINS, -50, 50, NBINS, -50, 50)
        h_RMS_GAN.Add(h_thetasqmean)
        h_RMS_GAN.Add(h_thetameansq)
        m = max(h_RMS_GAN.GetMaximum(), h_RMS_G4.GetMaximum())
        h_RMS_GAN.SetMaximum(m)
        h_RMS_G4.SetMaximum(m)
        
        hists_RMS[key] = [h_RMS_G4, h_RMS_GAN]
        hists_RMS[key][0].SetMaximum(MAX)
        hists_RMS[key][1].SetMaximum(MAX)
        hists_RMS[key][0].SetMinimum(MIN)
        hists_RMS[key][1].SetMinimum(MIN)
        hists_RMS[key][0].Write()
        hists_RMS[key][1].Write()
    fileout.Write()
