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


class dataset_stage1(torch.utils.data.Dataset):
    def __init__(self, X, img_size=256, root='', mode='Train'):

        super(dataset_stage1, self).__init__()

        self.root = root
        self.fns = X
        self.n_images = len(self.fns)   # number of HSIs
        self.indices = np.array(range(self.n_images))

        self.mode=mode
        self.img_size=img_size
    
        
    def __getitem__(self, index):
        data, label = {}, {}
        
        fn = os.path.join(self.root, self.fns[index])

        x = loadmat(fn)
        x = x[list(x.keys())[-1]]
         
        x = x.astype(np.float32)
        # x[x<0]=0
        xmin = np.min(x)
        xmax = np.max(x)
        
        if self.mode=='Train':

            flip = rn.random()
            if flip > 0 and flip < 0.25:
                x = x[::-1, :, :]
            if flip > 0.25 and flip < 0.5:
                x = x[:, ::-1, :]
            else:
                x = x
            x = torch.from_numpy(x.copy())

        elif self.mode == 'Validation':

            x = torch.from_numpy(x.copy())
        
        if xmin == xmax:
            print('nan in', self.fns[index])
            return np.zeros_like(x)
        
        x = (x-xmin) / (xmax-xmin)
        
        return x, fn

    def __len__(self):
        return self.n_images
    
