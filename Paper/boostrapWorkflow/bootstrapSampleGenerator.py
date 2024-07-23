import ROOT as R
import numpy as np
from array import array
import os
import json
from tqdm import tqdm

NSAMPLES = 1000 # number of samples per file
NIMAGES  = 2 # number of images per radius

DATA_DIR = '/home/ruben/Documents/DatosPipes/'

def getLocalIndex(globalIndex, nEntriesPerFile):
    _cumulativeEntries = np.cumsum(nEntriesPerFile)
    _file_idx = np.searchsorted(_cumulativeEntries, globalIndex)
    if _file_idx == 0 : _sample_idx = globalIndex
    else: _sample_idx = np.mod(globalIndex, _cumulativeEntries[_file_idx-1])
    # If last sample in file, jump to next
    if _sample_idx == nEntriesPerFile[_file_idx]:
        _file_idx += 1
        _sample_idx = 0
    return _file_idx, _sample_idx

if __name__=='__main__':

    with open('nEntries.json','r') as fdict:
        data = fdict.read()
    dataDict = json.loads(data)

    for key in dataDict:
        nEntries = []
        for f in dataDict[key]:
            nEntries.append(dataDict[key][f])
        nEntries = np.asarray(nEntries, dtype=int)
        nTotal = np.sum(nEntries)

        for i in range(NIMAGES):
            # Open file
            print(f'Creating file: Bootstrap_CosmicMuons_{key}_{i}.root')
            fileOut = R.TFile.Open(f'Bootstrap_CosmicMuons_{key}_{i}.root', 'RECREATE')
            treeOut = R.TTree(f"globalReco", f"globalReco")

            # Generate random numbers
            indexes = np.random.randint(low=0, high=nTotal, size=NSAMPLES, dtype=int)
            arrFileIdx, arrEntryIdx = [], []
            for idx in indexes:
                fileIdx, entryIdx = getLocalIndex(idx, nEntries)
                arrFileIdx.append(fileIdx)
                arrEntryIdx.append(entryIdx)
            arrIdx = np.asarray([arrFileIdx, arrEntryIdx], dtype=int)
            print(np.size(arrIdx), np.shape(arrIdx))

            # Containers for variables
            r = array('i', [0])
            px1 = array('f', [0])
            py1 = array('f', [0])
            pz1 = array('f', [0])
            pvx1 = array('f', [0])
            pvy1 = array('f', [0])
            pvz1 = array('f', [0])
            px2 = array('f', [0])
            py2 = array('f', [0])
            pz2 = array('f', [0])
            pvx2 = array('f', [0])
            pvy2 = array('f', [0])
            pvz2 = array('f', [0])
            # Branches
            treeOut.Branch("r", r, "r/I")
            treeOut.Branch("px1", px1, "px1/F")
            treeOut.Branch("py1", py1, "py1/F")
            treeOut.Branch("pz1", pz1, "pz1/F")
            treeOut.Branch("pvx1", pvx1, "pvx1/F")
            treeOut.Branch("pvy1", pvy1, "pvy1/F")
            treeOut.Branch("pvz1", pvz1, "pvz1/F")
            treeOut.Branch("px2", px2, "px2/F")
            treeOut.Branch("py2", py2, "py2/F")
            treeOut.Branch("pz2", pz2, "pz2/F")
            treeOut.Branch("pvx2", pvx2, "pvx2/F")
            treeOut.Branch("pvy2", pvy2, "pvy2/F")
            treeOut.Branch("pvz2", pvz2, "pvz2/F")
            # Get samples from root files
            for j,f in tqdm(enumerate(dataDict[key])):
                fileTmp = R.TFile.Open(f'{DATA_DIR}/{f}', 'READ')
                treeTmp = fileTmp.Get("globalReco")
                treeTmp.SetBranchAddress("px1", px1)
                #print(arrIdx[0,:])
                #print(np.shape(arrIdx[:,arrIdx[0,:]==j]))
                #print(arrIdx[:,arrIdx[0,:]==j][1,:])
                selectedIdx = arrIdx[:,arrIdx[0,:]==j]
                for entryToGet in selectedIdx[1,:]:
                    treeTmp.GetEntry(entryToGet)
                    r[0] = int(key)
                    px1[0]  = treeTmp.px1 
                    py1[0]  = treeTmp.py1
                    pz1[0]  = treeTmp.pz1
                    pvx1[0] = treeTmp.pvx1
                    pvy1[0] = treeTmp.pvy1
                    pvz1[0] = treeTmp.pvz1
                    px2[0]  = treeTmp.px2
                    py2[0]  = treeTmp.py2
                    pz2[0]  = treeTmp.pz2
                    pvx2[0] = treeTmp.pvx2
                    pvy2[0] = treeTmp.pvy2
                    pvz2[0] = treeTmp.pvz2
                    treeOut.Fill()
            fileOut.Write()
            fileOut.Close()
