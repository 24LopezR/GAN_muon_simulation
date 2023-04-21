import os
import numpy as np
import torch
from torch.utils.data import Dataset
import ROOT as r
from tqdm import tqdm
import csv

################### CONSTANTS #########################
DATAFILESPATH = '/home/ruben/Samples/training/csv/'
''' datafiles name: PipeTest_19p8_20_fullformat_19p8_20_seed20.root '''

#######################################################


class MuonDataset(Dataset):
    def __init__(self, datafiles_path, transform=None, target_transform=None):
        self.datafiles_path = datafiles_path
        self.transform = transform
        self.target_transform = target_transform

        # Dictionary with radius
        self.radius = {
            "18p0_20": 20,
            "18p2_20": 18,
            "18p4_20": 16,
            "18p6_20": 14,
            "18p8_20": 12,
            "19p0_20": 10,
            "19p2_20": 8,
            "19p4_20": 6,
            "19p6_20": 4,
            "19p8_20": 2
        }

        # we need to order the csv files that form the dataset somehow (alphabetic order)
        self.list_of_files = sorted([self.datafiles_path+_file for _file in os.listdir(self.datafiles_path)])
        self.samples_index = {}
        for _file in self.list_of_files:
            with open(_file) as f:
                self.samples_index[_file] = sum(1 for line in f)

    def __len__(self):
        length = 0
        for _file in os.listdir(self.datafiles_path):
            if '.csv' not in _file: continue
            with open(self.datafiles_path+_file) as f:
                l_temp = sum(1 for line in f)
            length += l_temp
        return length

    def __getitem__(self, idx):
        return 0

    def writeCSVfiles(self, datafiles_path):
        """
            Reads the muon data stored in .root files and prints the dataset as a csv
            file with the following format:
                - Size: [N, 9]
            [ x1_0, y1_0, vx1_0, vy1_0, Dx_0, Dy_0, Dvx_0, Dvy_0, r_0
              ...   ...   ...    ...    ...   ...   ...    ...    ...
              x1_N, y1_N, vx1_N, vy1_N, Dx_N, Dy_N, Dvx_N, Dvy_N, r_N ]
            ------------------------------------------------------------------------------
                                     TO BE USED ONLY ONCE !!!!!
            -----------------------------------------------------------------------------
        """
        # loop over files
        for name in os.listdir(datafiles_path):
            print('>> Processing file {0}'.format(name[:-5]))
            _data = []
            # determine radius of pipe
            for key in self.radius:
                if key in name:
                    r_pipe = self.radius[key]
            _f = r.TFile(datafiles_path + '/' + name)
            # loop over events
            for ev in tqdm(_f.globalReco, total=_f.globalReco.GetEntries(), desc='Loading data: ' + name):
                if ev.type1 != 3 or ev.type2 != 3:
                    continue
                if abs(ev.px1) > 80 or abs(ev.py1) > 80 or abs(ev.pvx1) > 1.5 or abs(ev.pvy1) > 1.5:
                    continue
                if abs(ev.px2) > 80 or abs(ev.py2) > 80 or abs(ev.pvx2) > 1.5 or abs(ev.pvy2) > 1.5:
                    continue
                _data.append([ev.px1, ev.py1, ev.pvx1, ev.pvy1, 
                    ev.px2 - ev.px1 + 39 * 2 * ev.pvx1, ev.py2 - ev.py1 + 39 * 2 * ev.pvy1, ev.pvx2 - ev.pvx1, ev.pvy2 - ev.pvy1, 
                    r_pipe])
            data = np.asarray(_data)
            np.savetxt("/home/ruben/Samples/training/{0}.csv".format(name[:-5]), data, delimiter=",")
            print('>> DONE: {0}'.format(name))
        print('Data successfully loaded')

    def getFileIndexFromIdx(self, idx):
        return idx

if __name__ == '__main__':
    data = MuonDataset(DATAFILESPATH)
    print(data.__len__())
    print(data.list_of_files)
    print(data.samples_index)
