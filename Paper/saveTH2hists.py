import ROOT as r
import numpy as np
import time

HIST_FILE = './rootFilesGen/PoCAmaps_gensamples2plus12mm.root'

if __name__=='__main__':
    #r.gROOT.ProcessLine('.L ./tdrstyle.C')
    r.gROOT.SetBatch(1)
    r.gStyle.SetOptStat(0)
    r.gStyle.SetCanvasBorderSize(0)
    r.gStyle.SetOptTitle(0)
    r.gStyle.SetPalette(r.kCMYK)
    f = r.TFile.Open(HIST_FILE, "READ")

    R = {'20': '18p0', 
         '18': '18p2', 
         '16': '18p4', 
         '14': '18p6',
         '12': '18p8',
         '10': '19p0',
         '8' : '19p2',
         '6' : '19p4',
         '4' : '19p6'}
    
    for rad in R:
        h = f.Get(f"gan_{rad}_RMS_GAN")
        c = r.TCanvas(f"c_{rad}_gan",f"c_{rad}_gan", 800, 800)
        c.GetPad(0).cd()
        c.SetLogz()  
        c.SetTopMargin(0.)
        c.SetBottomMargin(0.)
        c.SetRightMargin(0.)
        c.SetLeftMargin(0.)
        h.SetMaximum(1.)
        h.SetMinimum(1e-4)
        h.Draw('COL AH');
        time.sleep(2)
        c.GetPad(0).SaveAs(f"PoCAmaps/gan_{R[rad]}_RMS.png")
        del h 


