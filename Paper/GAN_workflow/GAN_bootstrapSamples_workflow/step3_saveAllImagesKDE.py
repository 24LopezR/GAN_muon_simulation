import ROOT as r
import numpy as np
import os
import time
from tqdm import tqdm
from PIL import Image

'''
Este script es el PASO 3 para sacar las imágenes de Geant4
Las guarda en formato .png
'''

HIST_FILE_PATH = '/home/ruben/Documents/GAN_muon_simulation/Paper/GAN_workflow/GAN_bootstrapSamples_workflow/2Dmaps/'
MAX_HIST = 0.0052
MIN_HIST = 0.

if __name__=='__main__':
    #r.gROOT.ProcessLine('.L ./tdrstyle.C')
    r.gROOT.SetBatch(1)
    r.gStyle.SetOptStat(0)
    r.gStyle.SetCanvasBorderSize(0)
    r.gStyle.SetOptTitle(0)
    r.gStyle.SetPalette(r.kCMYK)
    r.gStyle.SetPalette(r.kBird)

    for f in os.listdir(HIST_FILE_PATH):
        if not '_100' in f: continue
        file = r.TFile.Open(f'{HIST_FILE_PATH}/{f}', "READ")
        h = file.Get(f"hh_pdf__Y_Z")
        h.SetMaximum(MAX_HIST)
        h.SetMinimum(MIN_HIST)
        c = r.TCanvas(f"c_{f.split('.')[0]}",f"c_{f.split('.')[0]}", 100, 100)
        c.GetPad(0).cd()
        #c.SetLogz()
        c.SetTopMargin(0.)
        c.SetBottomMargin(0.)
        c.SetRightMargin(0.)
        c.SetLeftMargin(0.)
        h.Draw('COL AH');
        time.sleep(2)
        c.GetPad(0).SaveAs(f"/home/ruben/Documents/GAN_muon_simulation/Paper/GAN_workflow/GAN_bootstrapSamples_workflow/h_KDE_{f.split('.')[0]}_RMS.png")
        del h
        break
