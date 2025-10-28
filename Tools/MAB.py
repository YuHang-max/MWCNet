import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

#Gated Spatial Attention Unit (GSAU)
class GSAU(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        i_feats = n_feats * 2

        self.Conv1 = nn.Conv2d(n_feats, i_feats, 1, 1, 0)
        self.DWConv1 = nn.Conv2d(n_feats, n_feats, 3, 1, 1, groups=n_feats)
        self.Conv2 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)

        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

    def forward(self, x):
        shortcut = x.clone()

        x = self.Conv1(self.norm(x))
        a, x = torch.chunk(x, 2, dim=1)
        x = x * self.DWConv1(a)
        x = self.Conv2(x)

        return x * self.scale + shortcut

# multi-scale kernel attention (MKA)



class MKAl(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        if n_feats % 3 != 0:
            raise ValueError("n_feats must be divisible by 3 for MKA.")

        i_feats = 2 * n_feats

        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        self.LKA = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 3, 1, 1, groups=n_feats // 3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 3, stride=1, padding=4, groups=n_feats // 3, dilation=4),
            nn.Conv2d(n_feats // 3, n_feats // 3, 3, 1, 1)
        )

        self.X = nn.Conv2d(n_feats // 3, n_feats // 3, 3, 1, 1, groups=n_feats // 3)

        self.proj_first = nn.Sequential(
            nn.Conv2d(n_feats, i_feats, 3, 1, 1)
        )

        self.proj_last = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        )

    def forward(self, x):
        shortcut = x.clone()
        x = self.norm(x)
        x = self.proj_first(x)
        a, x = torch.chunk(x, 2, dim=1)
        a_1, a_2, a_3 = torch.chunk(a, 3, dim=1)
        a = torch.cat([self.LKA(a_1) * self.X(a_1), self.LKA(a_2) * self.X(a_2), self.LKA(a_3) * self.X(a_3)],
                      dim=1)
        x = self.proj_last(x * a) * self.scale + shortcut
        return x

class MKAm(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        if n_feats % 3 != 0:
            raise ValueError("n_feats must be divisible by 3 for MKA.")

        i_feats = 2 * n_feats

        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        self.LKA = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 3, 1, 1, groups=n_feats // 3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 3, stride=1, padding=4, groups=n_feats // 3, dilation=4),
            nn.Conv2d(n_feats // 3, n_feats // 3, 3, 1, 1)
        )

        self.X = nn.Conv2d(n_feats // 3, n_feats // 3, 3, 1, 1, groups=n_feats // 3)

        self.proj_first = nn.Sequential(
            nn.Conv2d(n_feats, i_feats, 3, 1, 1)
        )

        self.proj_last = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        )

    def forward(self, x):
        shortcut = x.clone()
        x = self.norm(x)
        x = self.proj_first(x)
        a, x = torch.chunk(x, 2, dim=1)
        a_1, a_2, a_3 = torch.chunk(a, 3, dim=1)
        a = torch.cat([self.LKA(a_1) * self.X(a_1), self.LKA(a_2) * self.X(a_2), self.LKA(a_3) * self.X(a_3)],
                      dim=1)
        x = self.proj_last(x * a) * self.scale + shortcut
        return x
class MKAs(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        if n_feats % 3 != 0:
            raise ValueError("n_feats must be divisible by 3 for MKA.")

        i_feats = 2 * n_feats

        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        self.LKA = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 3, 1, 1, groups=n_feats // 3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 3, stride=1, padding=4, groups=n_feats // 3, dilation=4),
            nn.Conv2d(n_feats // 3, n_feats // 3, 3, 1, 1)
        )

        self.X = nn.Conv2d(n_feats // 3, n_feats // 3, 3, 1, 1, groups=n_feats // 3)

        self.proj_first = nn.Sequential(
            nn.Conv2d(n_feats, i_feats, 3, 1, 1)
        )

        self.proj_last = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        )

    def forward(self, x):
        shortcut = x.clone()
        x = self.norm(x)
        x = self.proj_first(x)
        a, x = torch.chunk(x, 2, dim=1)
        a_1, a_2, a_3 = torch.chunk(a, 3, dim=1)
        a = torch.cat([self.LKA(a_1) * self.X(a_1), self.LKA(a_2) * self.X(a_2), self.LKA(a_3) * self.X(a_3)],
                      dim=1)
        x = self.proj_last(x * a) * self.scale + shortcut
        return x

#multi-scale attention blocks (MAB)
class MABS(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.SKA = MKAs(n_feats)
        self.SFE = GSAU(n_feats)

    def forward(self, x):
        x = self.SKA(x)
        x = self.SFE(x)
        return x

class MABM(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.SKA = MKAm(n_feats)
        self.SFE = GSAU(n_feats)

    def forward(self, x):
        x = self.SKA(x)
        x = self.SFE(x)
        return x

class MABL(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.SKA = MKAl(n_feats)
        self.SFE = GSAU(n_feats)

    def forward(self, x):
        x = self.SKA(x)
        x = self.SFE(x)
        return x

if __name__ == '__main__':
    n_feats = 3  # Must be divisible by 3
    block = MABS(n_feats)
    input = torch.randn(1, 3, 128, 128)
    output = block(input)
    print(input.size())
    print(output.size())