import pdb
from typing import Union, Optional
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from Tools.FCA import TripletAttention
from Tools.SGFN import SGFN
from Tools.MAB import *
from Tools.DSConv2d import DepthwiseSeparableConvWithWTConv2d
from Tools.FCM import SuperResolutionBlock
class ChannelAttention(nn.Module):
    """
    (N, C, H, W)            (N, C, 1, 1)      (N, C/λ, 1, 1)         (N, C, 1, 1)
     ---> (Spacial Average Pool) ---> Conv2d ---> ReLU ---> Conv2d ---> Sigmoid --->
    |                                                                             |
    x ------------------------------------------------------------------------->(*) ---> output
    """
    def __init__(self, in_channels: int, extract_coefficient: float = 8, num_bits: int = 32):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        assert 0 < num_bits <= 32
        if num_bits == 32:
            self.conv_extract = nn.Conv2d(
                in_channels, int(in_channels / extract_coefficient), kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)
            )
            self.conv_release = nn.Conv2d(
                int(in_channels / extract_coefficient), in_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)
            )
        else:
            self.conv_extract = QuantileConv2d(
                in_channels, int(in_channels / extract_coefficient), kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                num_bits=num_bits
            )
            self.conv_release = QuantileConv2d(
                int(in_channels / extract_coefficient), in_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                num_bits=num_bits
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 5:
            x_4d = to_4d_tensor(x)
            return to_5d_tensor(
                self.conv_release(F.leaky_relu(
                    self.conv_extract(self.avg_pool(x_4d)), negative_slope=0.2, inplace=True
                )), batch_size=x.shape[0]
            ) * x
        return self.conv_release(F.leaky_relu(
                    self.conv_extract(self.avg_pool(x)), negative_slope=0.2, inplace=True
                )) * x

class SpacialAttention(nn.Module):
    """
    (N, C, H, W)           (N, 1, H, W)     (N, 1, H, W)
     ---> (Channel Average Pool) ---> Conv2d ---> Sigmoid --->
    |                                                       |
    x --------------------------------------------------->(*) ---> output
    """
    def __init__(self, mid_channels: int = 1, num_bits: int = 32):
        super(SpacialAttention, self).__init__()
        self.avg_pool = self.SpacialAvgPool(mid_channels)
        assert 0 < num_bits <= 32
        if num_bits == 32:
            self.conv = nn.Conv2d(
                1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            )
        else:
            self.conv = QuantileConv2d(
                1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                num_bits=num_bits
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 5:
            x_4d = to_4d_tensor(x)
            return to_5d_tensor(
                F.sigmoid(torch.mean(
                    self.conv(self.avg_pool(x_4d)), dim=1, keepdim=True
                )), batch_size=x.shape[0]
            ) * x
        return F.sigmoid(torch.mean(
            self.conv(self.avg_pool(x)), dim=1, keepdim=True
        )) * x

    class SpacialAvgPool:
        """
        Used for calculating the average value within channels of the input tensor.
              _________      _______________
            / \________\          \                   when out_channels = 1
          /  / \________\    input channels                 ________
          \/  / \   ...  \          \           --->      /        /
           \/ ...\________\  ________\______            /________/
            \   /        /
             \/________/
        """

        def __init__(self, output_channels: int):
            self.output_channels = output_channels

        def __call__(self, x: torch.Tensor) -> Union[None, torch.Tensor]:
            y = None
            if len(x.shape) == 4:
                _n, _c, _h, _w = x.shape
                avg_interval = math.floor(_c / self.output_channels)
                y = torch.zeros(_n, self.output_channels, _h, _w).to(x.device)
                for i in range(self.output_channels):
                    if i == self.output_channels - 1:
                        y[:, i, :, :] = torch.mean(x[:, i * avg_interval:, :, :], dim=1)
                    else:
                        y[:, i, :, :] = torch.mean(x[:, i * avg_interval:(i + 1) * avg_interval, :, :], dim=1)
            elif len(x.shape) == 3:
                _c, _h, _w = x.shape
                avg_interval = math.floor(_c / self.output_channels)
                y = torch.zeros(self.output_channels, _h, _w).to(x.device)
                for i in range(self.output_channels):
                    if i == self.output_channels - 1:
                        y[i, :, :] = torch.mean(x[i * avg_interval:, :, :], dim=1)
                    else:
                        y[i, :, :] = torch.mean(x[i * avg_interval:(i + 1) * avg_interval, :, :], dim=1)
            return y

class TriSSA(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_bits: int = 32):
        """
         ---> Channel Attention --->                 ---> Spacial Attention --->
        |                         |                 |                         |
        x ---------------------> * ---> Conv2d ---> -----------------------> * ---> Conv2d ---> (activate) ---> y
        """
        super(TriSSA, self).__init__()
        self.TriSSA = TripletAttention()
        self.channel_attention = ChannelAttention(in_channels=in_channels, num_bits=num_bits)
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
        y = self.channel_compress(self.channel_attention(x))
        if self.num_bits == 32:
            return self.spacial_compress(self.spacial_attention(y)) ## torch.Size[1440, 27, 32, 1]
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
        #ys = ys[..., :-3]
        #average fusion
        y_fused = (ys + ym + yl) / 3
        interpolate = self.res_conv(F.interpolate(
            x, scale_factor=(self.step_scale), mode='bicubic'
        ))
        #y = self.spectral_superresolution1(self.mfa(y_fused))
        #print("interpolate:", interpolate.size())
        #pdb.set_trace()
        #y = self.spectral_superresolution1(self.mfa(y_fused))
        y = self.spectral_superresolution1(self.mfa(y_fused)) + interpolate # torch.Size([BS, 128, 64, 2])
        #print("y:", y.size())
        #pdb.set_trace()
        res = self.wt_superresolution(y)
        return self.srf(self.spatial_superresolution(res))

class MWCNet(nn.Module): ###编码和解码网络
    def __init__(self, num_bits: int = 32):
        super(MWCNet, self).__init__()
        self.encoder = TriSSA(in_channels=172, out_channels=27, num_bits=num_bits)
        self.decoder = MSWSS(in_channels=27, out_channels=172, scale=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x = self.prepro(x)
        return self.decoder(self.encoder(x))



if __name__ == '__main__':
    device = 'cuda:2'
    x = torch.randn(8, 172, 128, 4).to(device)
    # model = FocusCompressMethod(27, 172, scale=4).to(device)
    # model = FocusEncoder(172, 27).to(device)
    model = MWCNet(num_bits=32).to(device)
    total_param = sum([param.numel() for param in model.encoder.parameters()])
    print(total_param)
    print(model(x).shape)
    print(model(x))
    # print(model.encoder.spacial_attention.conv.weight)
    spacial_attention = SpacialAttention(num_bits=32).to(device)
    print(spacial_attention(x).shape)
    # for name, param in model.named_parameters():
    #     print(name)
    #     print(param.shape)
    #state = torch.load('./f2dcn_param/F2DCN_32_bit_12960_epoch.pth')
    state = torch.load('./f2dcnwy_param/F2DCN_32_bit_12960_epoch.pth')
    model.load_state_dict(state, strict=False)
    for key in state.keys():
        if 'encoder.spacial_attension' in key:
            print(key)
    model.encoder.spacial_attention.load_state_dict({
        'conv.weight':
            state['encoder.spacial_attension.conv.weight'],
        'conv.bias':
            state['encoder.spacial_attension.conv.bias'],
    })
