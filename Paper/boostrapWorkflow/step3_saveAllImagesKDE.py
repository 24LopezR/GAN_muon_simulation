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

HIST_FILE_PATH = '/home/ruben/Documents/GAN_muon_simulation/Paper/boostrapWorkflow/2Dmaps_KDE/'
MAX_HIST = 0.0052
MIN_HIST = 0.

if __name__=='__main__':
    r.gROOT.ProcessLine('.L ./tdrstyle.C')
    r.setTDRStyle()
    r.gROOT.SetBatch(1)
    r.gStyle.SetOptStat(0)
    #r.gStyle.SetCanvasBorderSize(0)
    r.gStyle.SetOptTitle(0)
    #r.gStyle.SetPalette(r.kCMYK)
    r.gStyle.SetPalette(r.kBird)
    r.gStyle.SetPadGridX(0)
    r.gStyle.SetPadGridY(0)

    for f in tqdm(os.listdir(HIST_FILE_PATH)[0:20]):
#        print(f)
        if os.path.exists(f"../PoCAmaps/h_KDE_{f.split('.')[0]}_RMS.png"): continue
        if not '16mm' in f: continue
        file = r.TFile.Open(f'{HIST_FILE_PATH}/{f}', "READ")
        h = file.Get(f"hh_pdf__Y_Z")
        h.SetMaximum(MAX_HIST)
        h.SetMinimum(MIN_HIST)
        c = r.TCanvas(f"c_{f.split('.')[0]}",f"c_{f.split('.')[0]}", 900, 800)
        c.cd()
        #c.SetLogz()
        #c.SetTopMargin(0.)
        #c.SetBottomMargin(0.)
        c.SetRightMargin(0.2)
        h.GetZaxis().SetTitleOffset(1.9)
        #c.SetLeftMargin(0.)
        h.Draw('COLZ');
        time.sleep(2)
        c.SaveAs(f"/home/ruben/Documents/GAN_muon_simulation/Paper/boostrapWorkflow/imagesKDE/h_KDE_{f.split('.')[0]}_forPaper.png")
        c.SaveAs(f"/home/ruben/Documents/GAN_muon_simulation/Paper/boostrapWorkflow/imagesKDE/h_KDE_{f.split('.')[0]}_forPaper.pdf")
        c.SaveAs(f"/home/ruben/Documents/GAN_muon_simulation/Paper/boostrapWorkflow/imagesKDE/h_KDE_{f.split('.')[0]}_forPaper.C")
        del h
