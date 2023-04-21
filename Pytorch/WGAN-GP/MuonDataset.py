import os
import numpy as np
import torch
from torch.utils.data import Dataset
import ROOT as r
from tqdm import tqdm

################### CONSTANTS #########################
DATAFILESPATH = '/home/ruben/Samples/training'
''' datafiles name: PipeTest_19p8_20_fullformat_19p8_20_seed20.root '''

#######################################################


class MuonDataset(Dataset):
    def __init__(self):

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

    def load(self, datafiles_path):
        """
            Reads the muon data files and returns the dataset as a numpy
            array with the following structure:
                - Size: [N, 9]
            [ x1_0, y1_0, vx1_0, vy1_0, Dx_0, Dy_0, Dvx_0, Dvy_0, r_0
              ...   ...   ...    ...    ...   ...   ...    ...    ...
              x1_N, y1_N, vx1_N, vy1_N, Dx_N, Dy_N, Dvx_N, Dvy_N, r_N ]
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
        

if __name__ == '__main__':
    MuonDataset().load(DATAFILESPATH)
