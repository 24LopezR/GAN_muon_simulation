import ROOT as R
import numpy as np
from array import array
import os
import json
from tqdm import tqdm

NSAMPLES = 300000 # number of samples per file
NIMAGES  = 10 # number of images per radius

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

'''
Reads a ROOT file with muon data and dumps the relevant branches
to a numpy array, to store it in memory
'''
def readROOTFile(ROOTfile):
    print(f'Reading file: {ROOTfile}')
    fileTmp = R.TFile.Open(ROOTfile, 'READ')
    treeTmp = fileTmp.Get("globalReco")
    data = []
    for i,e in tqdm(enumerate(treeTmp), total=treeTmp.GetEntries()):
        entryTmp = []
        entryTmp.append(int(i))
        entryTmp.append(e.px1)
        entryTmp.append(e.py1)
        entryTmp.append(e.pz1)
        entryTmp.append(e.pvx1)
        entryTmp.append(e.pvy1)
        entryTmp.append(e.pvz1)
        entryTmp.append(e.px2)
        entryTmp.append(e.py2)
        entryTmp.append(e.pz2)
        entryTmp.append(e.pvx2)
        entryTmp.append(e.pvy2)
        entryTmp.append(e.pvz2)
        entryTmp.append(e.type1)
        entryTmp.append(e.type2)
        data.append(entryTmp)
    fileTmp.Close()
    data = np.asarray(data, dtype=float)
    return data

if __name__=='__main__':

    with open('nEntries.json','r') as fdict:
        data = fdict.read()
    dataDict = json.loads(data)

    for key in dataDict:
        # Build didctionary with muon data. 'filename': 'numpy array with muons'
        muonDict = dict()
        nEntries = []
        for f in dataDict[key]:
            nEntries.append(dataDict[key][f])
            muonDict[f] = readROOTFile(f'{DATA_DIR}/{f}')
        nEntries = np.asarray(nEntries, dtype=int)
        nTotal = np.sum(nEntries)

        for i in range(NIMAGES):
            # Open file
            print(f'Creating file: DatosBootstrap/Bootstrap_CosmicMuons_{key}mm_{1000+i}.root')
            fileOut = R.TFile.Open(f'DatosBootstrap/Bootstrap_CosmicMuons_{key}mm_{1000+i}.root', 'RECREATE')
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
            event = array('i', [0])
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
            type1 = array('i', [0])
            type2 = array('i', [0])
            # Branches
            treeOut.Branch("r", r, "r/I")
            treeOut.Branch("event", event, "event/I")
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
            treeOut.Branch("type1", type1, "type1/I")
            treeOut.Branch("type2", type2, "type2/I")
            # Get samples from root files
            for j,f in enumerate(dataDict[key]):
                #fileTmp = R.TFile.Open(f'{DATA_DIR}/{f}', 'READ')
                #treeTmp = fileTmp.Get("globalReco")
                #print(arrIdx[0,:])
                #print(np.shape(arrIdx[:,arrIdx[0,:]==j]))
                #print(arrIdx[:,arrIdx[0,:]==j][1,:])
                selectedIdx = arrIdx[:,arrIdx[0,:]==j]
                for entryToGet in selectedIdx[1,:]:
                    entry = muonDict[f][entryToGet]
                    r[0] = int(key)
                    event[0]= int(entry[0])
                    px1[0]  = entry[1]
                    py1[0]  = entry[2]
                    pz1[0]  = entry[3]
                    pvx1[0] = entry[4]
                    pvy1[0] = entry[5]
                    pvz1[0] = entry[6]
                    px2[0]  = entry[7]
                    py2[0]  = entry[8]
                    pz2[0]  = entry[9]
                    pvx2[0] = entry[10]
                    pvy2[0] = entry[11]
                    pvz2[0] = entry[12]
                    type1[0] = int(entry[13])
                    type2[0] = int(entry[13])
                    treeOut.Fill()
            fileOut.Write()
            fileOut.Close()
        del muonDict
