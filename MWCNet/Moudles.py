import pdb
from typing import Union, Optional
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from Tools.FCA import *
from Tools.SGFN import SGFN
from Tools.MAB import *
from Tools.DSConv2d import DepthwiseSeparableConvWithWTConv2d
from Tools.FCM import SuperResolutionBlock

class TriSSA(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_bits: int = 32):
        super(TriSSA, self).__init__()
        self.TriSSA = TripletAttention()
        self.spectral_attention = ChannelAttention(in_channels=in_channels, num_bits=num_bits)
        self.spacial_attention = SpacialAttention(num_bits=num_bits)
        assert 0 < num_bits <= 32
        self.num_bits = num_bits

        if num_bits == 32:
            self.channel_compress = nn.Conv2d(
                in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            )
            self.spacial_compress = nn.Conv2d(
                out_channels, out_channels, kernel_size=(5, 4), stride=(4, 4), padding=(2, 0)
            )
        else:
            self.channel_compress = QuantileConv2d(
                in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                num_bits=num_bits
            )
            self.spacial_compress = QuantileConv2d(
                out_channels, out_channels, kernel_size=(5, 4), stride=(4, 4), padding=(2, 0),
                num_bits=num_bits
            )
            self.activate = QuantileActivate(num_bits=num_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.TriSSA(x)
        y = self.channel_compress(self.spectral_attention(x))
        if self.num_bits == 32:
            return self.spacial_compress(self.spacial_attention(y))
        else:
            return self.activate(
                self.spacial_compress(self.spacial_attention(y))
            )

class MSWSS(nn.Module):
    """
      ------------------------> Bicubic interpolate -------------------------->
     |                         ---> MFA Block -------->                       |
    |                         |                      |                      |
    x ---> Split channels ---> ---> MFA Block ---> Concat ---> SR Block ---> + ---> WT up scale ---> y
                            |                      |
                            ---> MFA Block -------->
                          |                      |
                         |          ...         |
                        |                      |
                        ---> MFA Block -------->
    """
    def __init__(
            self, in_channels: int, out_channels: int, scale: int, num_groups: int = 1, c1: int=64, c2: int=128,
            shared_params: bool = True
    ):
        super(MSWSS, self).__init__()
        if in_channels % num_groups != 0:
            raise ValueError(
                f'Input param in_channels: {in_channels} must be divided by num_groups: {num_groups}.'
            )
        channel_per_group = int(in_channels / num_groups)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shared_params = shared_params
        self.num_groups = num_groups
        self.step_scale = int(math.sqrt(scale))
        self.c1 = c1
        self.c2 = c2

        if shared_params:
            self.branch_1 = SuperResolutionBlock(in_channels=channel_per_group, scale=self.step_scale)
            self.mfas = MABS(3)
            self.mfam = MABM(9)
            self.mfal = MABL(self.in_channels)
        else:
            self.branch_1_list = []
            for _ in range(num_groups):
                self.branch_1_list.append(SuperResolutionBlock(
                    in_channels=channel_per_group, scale=self.step_scale
                ))
        self.spatial_superresolution = SuperResolutionBlock(in_channels=self.out_channels, scale=self.step_scale)
        self.res_conv = nn.Conv2d(
            self.in_channels, self.c2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.mfa = MKAs(self.in_channels)
        self.wt_superresolution = DepthwiseSeparableConvWithWTConv2d(in_channels=self.c2, out_channels=self.out_channels)
        self.spectral_superresolution1 = nn.Conv2d(
            self.in_channels, self.c2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.srf = SGFN(in_features=out_channels)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.branch_1(x)
        # large-scale
        yl = self.mfal(x1)
        #middle-scale
        group_intervalm = math.floor(self.in_channels / 3)
        _nm, _cm, _hm, _wm = x1.shape
        ym = torch.zeros(
            size=(_nm, _cm, _hm, _wm), device=x.device
        )
        for i in range(self.num_groups):
            xi = x1[:, i * group_intervalm: (i + 1) * group_intervalm, :, :]
            if self.shared_params:
                ym[:, i * group_intervalm: (i + 1) * group_intervalm, :, :] = self.mfam(xi)
            else:
                ym[:, i * group_intervalm: (i + 1) * group_intervalm, :, :] = self.mfam(xi)
        # small-scale
        group_intervals = math.floor(self.in_channels / 9)
        _ns, _cs, _hs, _ws = x1.shape
        ys = torch.zeros(
            size=(_ns, _cs, _hs, _ws), device=x.device
        )
        for i in range(self.num_groups):
            xi = x1[:, i * group_intervals: (i + 1) * group_intervals, :, :]
            if self.shared_params:
                ys[:, i * group_intervals: (i + 1) * group_intervals, :, :] = self.mfas(xi)
            else:
                ys[:, i * group_intervals: (i + 1) * group_intervals, :, :] = self.mfas(xi)
        y_fused = (ys + ym + yl) / 3
        interpolate = self.res_conv(F.interpolate(
            x, scale_factor=(self.step_scale), mode='bicubic'
        ))
        y = self.spectral_superresolution1(self.mfa(y_fused)) + interpolate

        res = self.wt_superresolution(y)
        return self.srf(self.spatial_superresolution(res))

class MWCNet(nn.Module):
    def __init__(self, num_bits: int = 32):
        super(MWCNet, self).__init__()
        self.encoder = TriSSA(in_channels=172, out_channels=27, num_bits=num_bits)
        self.decoder = MSWSS(in_channels=27, out_channels=172, scale=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x = self.prepro(x)
        return self.decoder(self.encoder(x))



