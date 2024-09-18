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
OUTPUT_FILE_PATH = "/home/ruben/Documents/PoCAmaps_50bins_KDE/"
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

    #numberOfColors = r.gStyle.GetNumberOfColors()
    #r.gStyle.SetPalette(r.kTemperatureMap)
    #print('Number of colors is', numberOfColors)
    # Create hists
    hists = [r.TH2F(f"h2_N",      f"Number of points YZ; Y; Z", NBINSX, -50, 50, NBINSY, -50, 50),
             r.TH2F(f"h2_theta",  f"Scattering angle YZ; Y; Z", NBINSX, -50, 50, NBINSY, -50, 50),
             r.TH2F(f"h2_theta2", f"Scattering angle squared YZ; Y; Z", NBINSX, -50, 50, NBINSY, -50, 50)]


    f = r.TFile.Open(f'{POCA_FILE_PATH}/{infilename}', "READ")
    tree = f.Get('tree')
    yc = array('d')
    zc = array('d')
    theta = array('d')
    zvar = r.RooRealVar("z","z", -50.0, 50.0)
    yvar = r.RooRealVar("y","y", -50.0, 50.0);
    d = r.RooDataSet("d","d", r.RooArgSet(zvar,yvar))
    le = [] # list of ellipses
    for ev in tree:
        if abs(ev.Y_PoCA) > 50 or abs(ev.Z_PoCA) > 50 or ev.theta < 0.1 or ev.theta > 0.3: continue
        yc.append(ev.Y_PoCA)
        zc.append(ev.Z_PoCA)
        theta.append(ev.theta)
        hists[0].Fill(ev.Y_PoCA, ev.Z_PoCA)
        hists[1].Fill(ev.Y_PoCA, ev.Z_PoCA, ev.theta)
        hists[2].Fill(ev.Y_PoCA, ev.Z_PoCA, ev.theta*ev.theta)
        yvar.setVal(ev.Y_PoCA)
        zvar.setVal(ev.Z_PoCA)
        d.add(r.RooArgSet(zvar,yvar))
        '''e = r.TEllipse(ev.Y_PoCA, ev.Z_PoCA, 5*ev.theta)
        e.SetFillColorAlpha(r.kBlack, 0.25)
        e.SetLineColorAlpha(r.kBlack, 0.25)
        le.append(e)'''
    f.Close()

    '''c = r.TCanvas("c","",800,800)
    c.cd()
    c.SetFillColor(r.kWhite)
    graph = r.TGraph(len(yc), yc, zc)
    graph.SetTitle("")
    graph.SetMarkerStyle(20)
    graph.SetMarkerSize(0.1)
    graph.GetXaxis().SetTitle("Y (cm)")
    graph.GetYaxis().SetTitle("Z (cm)")
    graph.Draw("AP")
    for e in le:
        e.Draw()
    c.SetTitle("")
    c.Update()
    c.SaveAs(f"c_{infilename.split('.')[0]}.png")
    c.SaveAs(f"c_{infilename.split('.')[0]}.pdf")'''

    kest4 = r.RooNDKeysPdf("kest4","kest4", r.RooArgSet(yvar,zvar), d)
    
    hh_data = d.createHistogram("hh_data", yvar, r.RooFit.Binning(NBINSX, -50.0, 50.0), r.RooFit.YVar(zvar, r.RooFit.Binning(NBINSY, -50.0, 50.0)))
    hh_pdf = kest4.createHistogram("hh_pdf", yvar, r.RooFit.Binning(NBINSX, -50.0, 50.0), r.RooFit.YVar(zvar, r.RooFit.Binning(NBINSY, -50.0, 50.0))) 
    
    '''c = r.TCanvas("rf707_kernelestimation", "rf707_kernelestimation", 1600, 800)
    c.Divide(2,1) 
    c.cd(1) 
    r.gPad.SetLeftMargin(0.15) 
    hh_data.GetZaxis().SetTitleOffset(1.4) 
    hh_data.SetTitle("Original measurement")
    hh_data.Draw("COLZ")
    c.cd(2)
    r.gPad.SetLeftMargin(0.20)
    hh_pdf.GetZaxis().SetTitleOffset(2.4) 
    hh_pdf.SetTitle("Smoothen measurement")
    hh_pdf.Draw("COLZ")
    c.SaveAs(f"plot_kernel_{infilename.split('.')[0]}.png")'''

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
    hh_data.Write()
    hh_pdf.Write()
    fileout.Write()
    fileout.Close()

if __name__== "__main__":
    r.gROOT.ProcessLine('.L ./tdrstyle.C')
    r.setTDRStyle()
    r.gROOT.SetBatch(1)
    if not os.path.exists(OUTPUT_FILE_PATH): os.makedirs(OUTPUT_FILE_PATH)
    for infilename in tqdm(listdir(POCA_FILE_PATH)):
        #print(f"{POCA_FILE_PATH}/{infilename}")
        if not '.root' in infilename: continue
        saveSingleHist(infilename)
