from SCVCN_BTCNet import *
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from dataset_stage1 import *
from dataset_ori import *
from trainOps import *
from RestoreCpmress import RestoreSkipConvCompress, DeviationHyperPriorRestoreSkipConvCompress
from CTCSN.model.CTCSN import CTCSN
from comparison_methods.model import DCSN
from comparison_methods.RSCC2 import RSCC2, RSCC3
from comparison_methods.BTCNet import BTCNet
import time
import numpy as np

batch_size = 1
device = torch.device('cuda:4')
MAX_EP = 6000
prefix = 'SCVCN'
VAL_HR =256
BANDS = 172
lmbda = 1e-2  # coefficient of bpp loss
SOURCE = 'stage1'
TARGET = 'stage1'
CR = 1

valfn = loadTxt('testpath/5HSI_test.txt')

val_loader = torch.utils.data.DataLoader(dataset_h5(valfn, mode='Validation',root=''), batch_size=1, shuffle=False, pin_memory=False)

# net = SCVCN_BTCNet_Fp32_GDN(cr=1).to(device)
# net = RSCC2(172, 27, 32, 32).to(device)
# net = RSCC2(172, 27, 32, 8).to(device)
quant_bit = 8
# net = RSCC3(172, 27, 32, quant_bit).to(device)
# net.load_state_dict(torch.load(f"./rscc_param/RSCC3_{quant_bit}bit_best.pth"))
# model_name = f"LiteRecon_{quant_bit}bit"

# for key, value in net.state_dict().items():
#     if "hyper_prior" in key:
#         print(key, )
# net = RestoreSkipConvCompress(32).to(device)
# state = torch.load("./rscc_param/RSCC2_best.pth")
# state = torch.load("./rscc_param/RSCC3_8bit_best.pth")
# net.load_state_dict(state)

# net = CTCSN(cr=1, bit_num=32).to(device)
# net.load_state_dict(torch.load("./ctcsn_param/CTCSN_33080_epoch.pth"))
# model_name = "CTCSN"

# net = DCSN(cr=1).to(device)
# net.load_state_dict(torch.load("./dcsn_param/RSCC_4880_epoch.pth"))
# model_name = "BTCNet"
# print(net.encoder.lamda_in)
# print(net.encoder.lamda_out)
# print(net.encoder.lamda_weight)
# print(net.encoder.lamda_bias)
net = BTCNet(bit_num=32).to(device)
net.load_state_dict(torch.load("./btc_param/BTCNet_best.pth"))
model_name = "DCSN"
def quantize(tensor_: torch.Tensor, quant_bit: int = 8) -> torch.Tensor:
    max_, min_ = torch.max(tensor_), torch.min(tensor_)
    quant_step = (max_ - min_) / (2 ** quant_bit - 1)
    tensor_ = (tensor_ - min_) / quant_step
    tensor_ = tensor_.round().to(torch.int)
    return tensor_ * quant_step + min_
# for param in net.encoder.parameters():
#     param.data = quantize(param.data)

# state = torch.load('checkpoint/SCVCN_stage1_stage1_cr_1_epoch_200.pth', map_location='cuda:1')
# net.load_state_dict(state)

with torch.no_grad():
    rmses, sams, fnames, psnrs = [], [], [], []

    for ind2, (vx, vfn) in enumerate(val_loader):
        net.eval()

        # start_time = time.time()
        vx = vx.view(vx.size()[0] * vx.size()[1], vx.size()[2], vx.size()[3], vx.size()[4])
        vx = vx.to(device).permute(0, 3, 1, 2).float()
        # y = net.encoder(vx)
        # y = quantize(y)
        # val_dec = net.decoder(y)
        start_time = time.time()
        # vx = vx.to(device).permute(0, 1, 4, 2, 3).float()
        y = net.encoder(vx)
        end_time = time.time()
        if isinstance(y, tuple):
            y = y[0]
        val_dec = net.decoder(y)
        if isinstance(val_dec, tuple):
            val_dec = val_dec[0]
        
        val_batch_size = len(vfn)
        # end_time = time.time()
        # print((end_time - start_time) / val_batch_size)
        
        # val_batch_size = len(vfn)
        img = [np.zeros((VAL_HR, VAL_HR, BANDS)) for _ in range(val_batch_size)]
        val_dec = val_dec.permute(0, 2, 3, 1).cpu().numpy()
        cnt = 0

        for bt in range(val_batch_size):
            for z in range(0, VAL_HR, 4):
                img[bt][:, z:z + 4, :] = val_dec[cnt]
                cnt += 1
            save_path = vfn[bt].split('/')
            # save_path = save_path[-2] + '-' + save_path[-1]
            # np.save('Rec/%s.npy' % (save_path), val_dec[bt])
            # end_time = time.time()
            print(end_time - start_time)
            GT = lmat(vfn[bt]).astype(np.float32)
            maxv, minv = np.max(GT), np.min(GT)
            img[bt] = img[bt] * (maxv - minv) + minv  ## De-normalization
            # np.save(f"./npys/{model_name}_{save_path[-1][:-4]}.npy", img[bt])
            
            # print(np.std(img[bt]))
            sams.append(sam(GT, img[bt]))
            rmses.append(rmse(GT, img[bt]))
            fnames.append(save_path)
            psnrs.append(psnr(img[bt], GT))
print(sum(psnrs) / len(psnrs))
print(psnrs)
print(rmses)
print(sams)
# print('\n\n %.3f/%.3f/%.3f,  bit rate:%.3f , time:%.3f' %
#           (np.mean(sams), np.mean(rmses), np.mean(psnrs), bpppb, end_time-start_time/val_batch_size))
