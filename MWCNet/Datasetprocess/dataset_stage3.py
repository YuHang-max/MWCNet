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


class dataset_stage3(torch.utils.data.Dataset):
    def __init__(self, X, img_size=256, crop_size=128, width=4, dataset_name='KSC', root='', mode='Train'):

        super(dataset_stage3, self).__init__()

        self.root = root
        self.fns = X
        self.n_images = len(self.fns)
        self.indices = np.array(range(self.n_images))

        self.mode=mode
        self.crop_size=crop_size
        self.img_size=img_size
        self.width = width
    
        
    def __getitem__(self, index):
        data, label = {}, {}
        
        fn = os.path.join(self.root, self.fns[index])

        x=loadmat(fn)
        x=x[list(x.keys())[-1]]
         
        x = x.astype(np.float32)
#         x[x<0]=0
        xmin = np.min(x)
        xmax = np.max(x)
        
        if self.mode=='Train':
            flip = rn.random()
            xx = []
            for k in range(0, x.shape[1], self.width):  # 32 for KSC, 16 for Salinas
                y = x[:, k:k+self.width, :]

                # if flip > 0 and flip < 0.25:
                #     y = y[::-1, :, :]
                # if flip > 0.25 and flip < 0.5:
                #     y = y[:, ::-1, :]

                y = torch.from_numpy(y.copy())
                xx.append(y)
            x = torch.stack(xx)
        else:
            xx = []
            for k in range(0, x.shape[1], self.width):  # 32 for KSC, 16 for Salinas
                y = x[:, k:k+self.width, :]
                y = torch.from_numpy(y.copy())
                xx.append(y)
            x = torch.stack(xx)

        if xmin == xmax:
            print('nan in', self.fns[index])
            return np.zeros((100, 512, 4, 172))
        
        x = (x-xmin) / (xmax-xmin)
        
        return x, fn

    def __len__(self):
        return self.n_images
    
