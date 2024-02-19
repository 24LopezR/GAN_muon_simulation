import numpy as np
import ROOT as r
import os
from os import listdir
from tqdm import tqdm
from array import array
import math

'''
Este script es el PASO 1 para obtener las imágenes con los datos de G4
Lee los archivos .root de G4, calcula el PoCA, y lo guarda en otro archivo .root

Para cada file llama a "runPoCAonSingleFile.py"
'''

ABSPATH = '/'.join(__file__.split('/')[:-2])
print('Project absolute path: ', ABSPATH)

DATA_SAMPLES_PATH = '/home/ruben/Documents/DatosPipes/'
OUTPUT_PATH       = '/home/ruben/Documents/PoCA_G4/'

if __name__== "__main__":
    if not os.path.exists(OUTPUT_PATH): os.makedirs(OUTPUT_PATH)
    for infilename in listdir(DATA_SAMPLES_PATH):
        if not '.root' in infilename: continue
        command = f'python3 runPoCAonSingleFile.py -i {infilename} >logs/log_{infilename.split(".")[0]}'
        print(command)
        os.system(command)
