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


class dataset_Cdata(torch.utils.data.Dataset):
    def __init__(self, X, img_size=512, crop_size=512, width=4, root='', mode='Train', dataset_name='KSC', marginal=2):

        super(dataset_Cdata, self).__init__()

        self.root = root
        self.fns = X
        self.n_images = len(self.fns)
        self.indices = np.array(range(self.n_images))
        self.dataset_name = dataset_name
        self.mode=mode
        self.crop_size=crop_size
        self.img_size=img_size
        self.width = width
        self.ynum = round(self.img_size/self.crop_size)

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
            self.xim = round(x.shape[1]/self.width)  # 8 for KSC, 2 for Salinas
            # Random crop
            xx = []
            for i in range(self.ynum):  # 4 for 128, 2 for 256
                for k in range(self.xim):
                    y = x[self.crop_size * i: self.crop_size * (i + 1),
                        self.width * k: self.width * (k + 1), :]
                    # Random flip
                    # flip = rn.random()
                    # if flip > 0 and flip < 0.25:
                    #     y = y[::-1, :, :]
                    # if flip > 0.25 and flip < 0.5:
                    #     y = y[:, ::-1, :]

                    y = torch.from_numpy(y.copy())
                    xx.append(y)
            x = torch.stack(xx)
        else:
            xx = []
            for k in range(0, x.shape[1], self.width):
                y = x[:, k:k+self.width, :]
                y = torch.from_numpy(y.copy())
                xx.append(y)
            x = torch.stack(xx)

        
        if xmin == xmax:
            print('nan in', self.fns[index])
            return np.zeros((self.marginal, 128, 4, 172))
        
        x = (x-xmin) / (xmax-xmin)
        
        return x, fn

    def __len__(self):
        return self.n_images
    
