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
MAX_HIST = 1.
MIN_HIST = 1e-4
NBINSX = 50
NBINSY = 50
###############################################################################

ABSPATH = '/'.join(__file__.split('/')[:-2])
print('Project absolute path: ', ABSPATH)

def saveSingleHist(infilename):
    # Create hists
    hists = [r.TH2F(f"h2_N",      f"Number of points YZ; Y; Z", NBINSX, -50, 50, NBINSY, -50, 50),
             r.TH2F(f"h2_theta",  f"Scattering angle YZ; Y; Z", NBINSX, -50, 50, NBINSY, -50, 50),
             r.TH2F(f"h2_theta2", f"Scattering angle squared YZ; Y; Z", NBINSX, -50, 50, NBINSY, -50, 50)]

    f = r.TFile.Open(f'{POCA_FILE_PATH}/{infilename}', "READ")
    tree = f.Get('tree')
    yc = array('d')
    zc = array('d')
    theta = array('d')
    for ev in tree:
        if abs(ev.Y_PoCA) > 70 or abs(ev.Z_PoCA) > 70 or ev.theta < 0.15 or ev.theta > 0.3: continue
        yc.append(ev.Y_PoCA)
        zc.append(ev.Z_PoCA)
        theta.append(ev.theta)
        hists[0].Fill(ev.Y_PoCA, ev.Z_PoCA)
        hists[1].Fill(ev.Y_PoCA, ev.Z_PoCA, ev.theta)
        hists[2].Fill(ev.Y_PoCA, ev.Z_PoCA, ev.theta*ev.theta)
    f.Close()

    c = r.TCanvas("c","",800,800)
    c.cd()
    c.SetFillColor(r.kWhite)
    graph = r.TGraph(len(yc), yc, zc)
    graph.SetTitle("")
    graph.SetMarkerStyle(20)
    graph.SetMarkerSize(0.1)
    '''graph.GetXaxis().SetAxisColor(r.kWhite)
    graph.GetYaxis().SetAxisColor(r.kWhite)
    graph.GetXaxis().SetLabelColor(r.kWhite)
    graph.GetYaxis().SetLabelColor(r.kWhite)'''
    graph.GetXaxis().SetTitle("Y (cm)")
    graph.GetYaxis().SetTitle("Z (cm)")
    graph.Draw("AP")
    le = []
    for i in range(len(yc)):
        e = r.TEllipse(yc[i], zc[i], 5*theta[i])
        e.SetFillColorAlpha(r.kBlue+3, 0.25)
        e.SetLineColorAlpha(r.kBlue+3, 0.25)
        e.Draw()
        le.append(e)
    c.SetTitle("")
    c.Update()

    c.SaveAs(f"c_{infilename.split('.')[0]}.png")
    c.SaveAs(f"c_{infilename.split('.')[0]}.pdf")

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
    r.gROOT.ProcessLine('.L ./tdrstyle.C')
    r.setTDRStyle()
    r.gROOT.SetBatch(1)
    if not os.path.exists(OUTPUT_FILE_PATH): os.makedirs(OUTPUT_FILE_PATH)
    for infilename in listdir(POCA_FILE_PATH):
        if not '.root' in infilename: continue
        if not '_100' in infilename: continue
        saveSingleHist(infilename)
        break
