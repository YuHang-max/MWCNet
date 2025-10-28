import glob
import numpy as np
import torch, os, pdb
import random as rn
from torch.utils import data
from torchvision import transforms
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
import torch.multiprocessing
from PIL import Image
from scipy.io import loadmat


class gt_stage3(torch.utils.data.Dataset):
    def __init__(self, gt, root=''):

        super(gt_stage3, self).__init__()

        self.root = root
        self.fns = gt
        self.n_images = len(self.fns)
        self.indices = np.array(range(self.n_images))

        
    def __getitem__(self, index):
        data, label = {}, {}
        
        fn = os.path.join(self.root, self.fns[index])

        g=loadmat(fn)
        g=g[list(g.keys())[-1]]
        
        return g, fn

    def __len__(self):
        return self.n_images
    
