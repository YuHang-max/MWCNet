import os.path
from typing import Union
import numpy as np
import random

import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat


def loadTxt(txt_path: str):
    lines = []
    with open(txt_path, 'r') as file:
        data = file.readlines()
        for item in data:
            lines.append(item.strip('\n'))
    return lines


def reshapeTrainData(x: torch.Tensor) -> torch.Tensor:
    _n, _s, _h, _w, _c = x.shape
    return x.reshape(-1, _h, _w, _c).permute(0, 3, 1, 2)


def reshapeTestData(x: torch.Tensor, batch_size: int) -> torch.Tensor:
    _n_s, _c, _h, _w = x.shape
    x = x.permute(0, 2, 3, 1)
    slice = x.reshape(batch_size, -1, _h, _w, _c).shape[1]
    y = torch.zeros(batch_size, _h, _w * slice, _c).to(x.device)
    for i in range(batch_size):
        for j in range(_w):
            y[i, :, j * _w: (j + 1) * _w, :] = x[i * _w + j]
    return y


def sam(y: torch.Tensor, x: torch.Tensor) -> float:
    x, y = x.cpu().detach().numpy(), y.cpu().detach().numpy()
    num = np.sum(np.multiply(x, y), 2)
    den = np.sqrt(np.multiply(np.sum(x**2, 2), np.sum(y**2, 2)))
    sam = np.sum(np.degrees(np.arccos(num / den))) / (x.shape[0]*x.shape[1])
    return sam


def rmse(y: torch.Tensor, x: torch.Tensor) -> float:
    x, y = x.cpu().detach().numpy(), y.cpu().detach().numpy()
    aux = np.sum(np.sum((x-y)**2, 0),0) / (x.shape[0]*x.shape[1])
    rmse_per_band = np.sqrt(aux)
    rmse_total = np.sqrt(np.sum(aux) / x.shape[2])
    return rmse_total


def psnr(y: torch.Tensor, x: torch.Tensor) -> float:
    x, y = x.cpu().detach().numpy(), y.cpu().detach().numpy()
    bands = x.shape[2]
    x = np.reshape(x, [-1, bands])
    y = np.reshape(y, [-1, bands])
    msr = np.mean((x-y)**2, 0)
    maxval = np.max(y, 0)**2
    return np.mean(10*np.log10(maxval/msr))


class BaseDataset(Dataset):
    def __init__(
            self, txt_path: str, root: Union[str, list] = '', crop_size: Union[int, list, tuple] = (128, 4 + 60 - 1),
            width: int = 4, train: bool = True
    ):
        super(BaseDataset, self).__init__()
        self.file_names = loadTxt(txt_path)
        self.roots = [root] if isinstance(root, str) else root
        self.crop_size = (crop_size, crop_size) if isinstance(crop_size, int) else crop_size
        self.width = width
        self.train = train

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index: int):
        item = dict()
        file_name = self.file_names[index]
        data = None
        for root in self.roots:
            file_path = os.path.join(root, file_name)
            if os.path.isfile(file_path):
                data_dict = loadmat(file_path)
                data = data_dict[list(data_dict.keys())[-1]].astype(np.float32)
                break
        _h, _w, _c = data.shape
        x_min, x_max = np.min(data), np.max(data)
        if x_min == x_max:
            return self.__getitem__(random.randint(0, self.__len__()))
        y = []
        if self.train:
            min_h, min_w = random.randint(0, _h - self.crop_size[0]), random.randint(0, _w - self.crop_size[1])
            for i in range(self.crop_size[1] - self.width + 1):
                xi = data[min_h: min_h + self.crop_size[0], min_w + i: min_w + i + self.width, :]
                flip = random.random()
                if 0 < flip < 0.25:
                    xi = xi[::-1, :, :]
                elif 0.25 < flip < 0.5:
                    xi = xi[:, ::-1, :]
                y.append(torch.from_numpy(xi.copy()))
            y = torch.stack(y)
            item['train_data'] = (y - x_min) / (x_max - x_min)
        else:
            for i in range(0, data.shape[1], self.width):
                xi = data[:, i:i+self.width, :]
                y.append(torch.from_numpy(xi.copy()))
            y = torch.stack(y)
            item['test_data'] = (y - x_min) / (x_max - x_min)
            item['GT'] = data
        return item


if __name__ == '__main__':
    dataset = BaseDataset(
        '../testpath/5HSI_test.txt', root=['./dataset/AVIRIS/train/hsi', './dataset/AVIRIS/test/hsi'],
        train=False
    )
    # print(dataset[0]['GT'].shape)
    dataloader = DataLoader(dataset=dataset, batch_size=12, shuffle=True)
    for item in dataloader:
        print(item['test_data'])
    x = torch.zeros(5 * 64, 172, 256, 4)
    print(reshapeTestData(x, batch_size=5).shape)
    x = torch.zeros(5, 60, 128, 4, 172)
    print(reshapeTrainData(x).shape)
