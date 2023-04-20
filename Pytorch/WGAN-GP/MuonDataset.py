import os
import numpy as np
import torch
from torch.utils.data import Dataset

class MuonDataset(Dataset):
    def __init__(self, 
