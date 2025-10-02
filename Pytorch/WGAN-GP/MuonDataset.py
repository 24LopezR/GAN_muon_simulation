import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
#import ROOT as r
from tqdm import tqdm
import csv
import pandas as pd
import linecache

################### CONSTANTS #########################
DATAFILESPATH = '/home/ruben/Documents/Samples_csv/'
''' datafiles name: PipeTest_19p8_20_fullformat_19p8_20_seed20.root '''
VARIABLES = ['x1', 'y1', 'vx1', 'vy1', 'Dx*', 'Dy*', 'Dvx', 'Dvy']
DATASET_MEANS = [-3.35106871e-03, 8.09916705e-04, -5.06725966e-05, -2.52541321e-05, 1.50628691e-03, -3.78948043e-04, -2.91726292e-05, 6.84416923e-06, 0.]
DATASET_STDS = [21.68090625, 21.68617845, 0.23085455, 0.23050012, 1.16791604, 1.17080322, 0.02676904, 0.0266398, 1.]
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
        self.radius_v = np.array([2., 4., 6., 8., 10., 12., 14., 16., 18., 20.])

        # we need to order the csv files that form the dataset somehow (alphabetic order)
        self.list_of_files = sorted([self.datafiles_path+_file for _file in os.listdir(self.datafiles_path)])
        num_samples = []
        for _file in self.list_of_files:
            with open(_file) as f:
                num_samples.append(sum(1 for line in f))
        self.num_samples = np.asarray(num_samples, dtype=int)
        self.cumulative_samples = np.cumsum(self.num_samples, dtype=int)
        #### Scaling constants
        self.means = torch.tensor(DATASET_MEANS)
        self.stds  = torch.tensor(DATASET_STDS)
        self.epsilon = 1e-07

    def __len__(self):
        length = 0
        for _file in self.list_of_files:
            with open(_file) as f:
                l_temp = sum(1 for line in f)
            length += l_temp
        return length

    def __getitem__(self, idx):
        _file_idx, _sample_idx = self.getFileIndexFromIdx(idx)
        _line = linecache.getline(self.list_of_files[_file_idx], _sample_idx+1)
        _sample = torch.tensor(np.fromstring(_line, sep=','), dtype=torch.float)
        if _sample.shape != torch.Size([9]):
            print(_sample.shape)
            print(_line)
            print(_file_idx, _sample_idx)
            print(_sample)
        _scaled_sample = (_sample - self.means) / (self.stds + self.epsilon)
        return _scaled_sample[0:4], _scaled_sample[4:8], self.oneHotEncode(_scaled_sample[8])

    def oneHotEncode(self, radius):
        return F.one_hot(torch.tensor(np.where(np.isclose(radius, self.radius_v))[0]), num_classes = self.radius_v.size).view(self.radius_v.size).type(dtype=torch.float)


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
        _file_idx = np.searchsorted(self.cumulative_samples, idx)
        #print('>> idx = {0}, cum_samples = {1}'.format(idx, self.cumulative_samples[_file_idx-1]))
        # Compute the sample id number
        if _file_idx == 0: _sample_idx = idx
        else: _sample_idx = np.mod(idx, self.cumulative_samples[_file_idx-1])
        # If last sample in file, jump to next file
        if _sample_idx == self.num_samples[_file_idx]:
            _file_idx += 1
            _sample_idx = 0
        #print('>> File index:   {0}'.format(_file_idx))
        #print('>> Sample index: {0}'.format(_sample_idx))
        return _file_idx, _sample_idx

    def fitScaler(self):
        _data = None
        for _f in self.list_of_files:
            _d = np.loadtxt(_f, delimiter=',')
            if _data is not None: _data = np.concatenate((_data, _d), axis=0)
            else: _data = _d
        means = _data.mean(axis=0)
        stds = _data.std(axis=0)
        print(means[0:7])
        print(stds[0:7])
        return means[0:7], stds[0:7]


if __name__ == '__main__':
    data = MuonDataset(DATAFILESPATH)
    print(data.__len__())
    print(data.num_samples)
    print(data.cumulative_samples)
    #for i in np.random.randint(low=0, high=data.__len__(), size=2):
    #    print(i)
    #    print(data.__getitem__(i))
    print(data.__getitem__(0))
    print(data.__getitem__(1842678))
    print(data.__getitem__(data.__len__()-1))
