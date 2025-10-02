import numpy as np
import ROOT as R

if __name__=="__main__":
    R.gROOT.ProcessLine('.L ./tdrstyle.C')
    R.gROOT.SetBatch(1)
    R.setTDRStyle()
    R.gStyle.SetErrorX(0.)
    R.gStyle.SetOptFit(0)
    
    x  = np.array([100,1e3,1e4,1e5,5e5,1e6,2e6])
    t1 = np.array([3.900,4.375,3.792,4.782,4.615,5.371,7.409])
    t2 = np.array([3.777,3.794,3.989,4.396,4.622,5.402,7.185])
    t3 = np.array([3.793,4.306,3.908,4.057,4.598,5.866,7.101])
    t = np.mean([t1,t2,t3], axis=0)
    print(t)

    a = np.array([1e3,1e4,1e5,5e5,1e6,2e6])
    b = np.array([0.456, 1.545, 13.27, 60+2.165, 120+3.59,4*60+10.9])

    h1 = R.TGraph(len(x[1:]), x[1:], t[1:])
    h2 = R.TGraph(len(a), a, b)
    m = R.TMultiGraph()
    m.SetTitle(";Number of samples;User time (s)")
  
    f1 = R.TF1("f1","pol1")
    f2 = R.TF1("f2","pol1")
 
    c = R.TCanvas("c","", 800, 800)
    h1.SetMarkerColor(R.kBlue+1)
    f1.SetLineColor(R.kBlue+1)
    h1.Fit("f1","","",1000.,2e6)
    h2.SetMarkerColor(R.kRed+1)
    f2.SetLineColor(R.kRed+1)
    h2.Fit("f2","","",1000.,2e6)
    c.SetLogx()
    c.SetLogy()
    m.Add(h1)
    m.Add(h2)
    m.SetMaximum(260.)
    m.Draw("AP")

    l = R.TLegend(0.2,0.7,0.4,0.83)
    l.AddEntry(h2, "GEANT4 setup", "P")
    l.AddEntry(h1, "cGAN setup", "P")
    l.SetFillStyle(0)
    l.SetTextFont(42)
    l.SetTextSize(0.025)
    l.SetBorderSize(0)
    l.Draw()

    c.SaveAs("timePlot.png")
    c.SaveAs("timePlot.pdf")
    c.SaveAs("timePlot.C")
