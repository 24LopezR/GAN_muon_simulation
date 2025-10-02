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
POCA_FILE_PATH = '/home/ruben/Documents/GAN_muon_simulation/Paper/GAN_workflow/GAN_bootstrapSamples_workflow/DatosGANPoCA/'
OUTPUT_FILE_PATH = "/home/ruben/Documents/GAN_muon_simulation/Paper/GAN_workflow/GAN_bootstrapSamples_workflow/2Dmaps/"
MAX_HIST = 0.
MIN_HIST = 1.
NBINSX = 50
NBINSY = 50
XLIM = 35
YLIM = 35
###############################################################################

ABSPATH = '/'.join(__file__.split('/')[:-2])
print('Project absolute path: ', ABSPATH)

def saveSingleHist(infilename):
    # Create hists
    hists = [r.TH2F(f"h2_N",      f"Number of points YZ; Y; Z", NBINSX, -XLIM, XLIM, NBINSY, -YLIM, YLIM),
             r.TH2F(f"h2_theta",  f"Scattering angle YZ; Y; Z", NBINSX, -XLIM, XLIM, NBINSY, -YLIM, YLIM),
             r.TH2F(f"h2_theta2", f"Scattering angle squared YZ; Y; Z", NBINSX, -XLIM, XLIM, NBINSY, -YLIM, YLIM)]

    f = r.TFile.Open(f'{POCA_FILE_PATH}/{infilename}', "READ")
    tree = f.Get('tree')
    zvar = r.RooRealVar("Z","Z (cm)", -YLIM, YLIM)
    yvar = r.RooRealVar("Y","Y (cm)", -XLIM, XLIM);
    d = r.RooDataSet("d","d", r.RooArgSet(zvar,yvar))
    for ev in tree:
        if abs(ev.Y_PoCA) > XLIM or abs(ev.Z_PoCA) > YLIM or ev.theta < 0.1 or ev.theta > 0.3: continue
        hists[0].Fill(ev.Y_PoCA, ev.Z_PoCA)
        hists[1].Fill(ev.Y_PoCA, ev.Z_PoCA, ev.theta)
        hists[2].Fill(ev.Y_PoCA, ev.Z_PoCA, ev.theta*ev.theta)
        yvar.setVal(ev.Y_PoCA)
        zvar.setVal(ev.Z_PoCA)
        d.add(r.RooArgSet(zvar,yvar))
    f.Close()

    kest4 = r.RooNDKeysPdf("kest4","kest4", r.RooArgSet(yvar,zvar), d)
    
    hh_data = d.createHistogram("hh_data", yvar, r.RooFit.Binning(NBINSX, -XLIM, XLIM), r.RooFit.YVar(zvar, r.RooFit.Binning(NBINSY, -YLIM, YLIM)))
    hh_pdf = kest4.createHistogram("hh_pdf", yvar, r.RooFit.Binning(NBINSX, -XLIM, XLIM), r.RooFit.YVar(zvar, r.RooFit.Binning(NBINSY, -YLIM, YLIM)))
    
    '''c = r.TCanvas("rf707_kernelestimation", "rf707_kernelestimation", 1800, 800)
    c.Divide(2,1) 
    c.cd(1) 
    r.gPad.SetLeftMargin(0.15)
    r.gStyle.SetPadGridX(0)
    r.gStyle.SetPadGridY(0)
    hh_data.GetZaxis().SetTitleOffset(1.4) 
    hh_data.SetTitle("Original measurement")
    hh_data.Draw("COLZ")
    c.cd(2)
    r.gPad.SetLeftMargin(0.20)
    r.gStyle.SetPadGridX(0)
    r.gStyle.SetPadGridY(0)
    hh_pdf.GetZaxis().SetTitleOffset(2.4) 
    hh_pdf.SetTitle("Smoothen measurement")
    hh_pdf.Draw("COLZ")
    c.SaveAs(f"plot_kernel_{infilename.split('.')[0]}.png")'''

    global MAX_HIST, MIN_HIST
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
    h_RMS = r.TH2F(f"h2_RMS", f"Scattering angle RMS YZ; Y; Z", NBINSX, -XLIM, XLIM, NBINSY, -YLIM, YLIM)
    h_RMS.Add(h_thetasqmean)
    h_RMS.Add(h_thetameansq)
    #h_RMS.SetMaximum(MAX_HIST)
    #h_RMS.SetMinimum(MIN_HIST)
    h_RMS.Write()
    h_RMS_a = r.TH2F(f"h2_RMS_a", f"Scattering angle RMS (approx) YZ; Y; Z", NBINSX, -XLIM, XLIM, NBINSY, -YLIM, YLIM)
    h_RMS_a = h_thetasqmean
    h_RMS_a.Write()
    hh_data.Write()
    hh_pdf.Write()
    if hh_pdf.GetMaximum() > MAX_HIST:
        MAX_HIST = hh_pdf.GetMaximum()
    if hh_pdf.GetMinimum() < MIN_HIST:
        MIN_HIST = hh_pdf.GetMinimum()
    fileout.Write()
    fileout.Close()

if __name__== "__main__":
    r.gROOT.ProcessLine('.L ./tdrstyle.C')
    r.setTDRStyle()
    r.gROOT.SetBatch(1)
    if not os.path.exists(OUTPUT_FILE_PATH): os.makedirs(OUTPUT_FILE_PATH)
    for infilename in tqdm(listdir(POCA_FILE_PATH)):
        if not '.root' in infilename: continue
        if not '_100' in infilename: continue
        saveSingleHist(infilename)
    print(MAX_HIST, MIN_HIST)
