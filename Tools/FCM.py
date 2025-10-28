from typing import Union, Optional
import math
import torch
from torch import nn
from torch.nn import functional as F

from .SampleTransformer import QuantileConv2d, QuantileActivate


def to_4d_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    Transform 5d tensor to 4d shape, where:
        _n: batch size,
        _c: channels that are used for Conv3d computation,
        _d, _h, _w: 3d spacial shape of input image
    """
    assert len(x.shape) == 5
    _n, _c, _d, _h, _w = x.shape
    return x.transpose(1, 2).reshape(-1, _c, _h, _w)


def to_5d_tensor(x: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    Transform 4d tensor to 5d shape, where:
        _n: batch size multiplies _d,
        _c: channels that are used for Conv3d computation,
        _d, _h, _w: 3d spacial shape of input image
    """
    assert len(x.shape) == 4
    _n, _c, _h, _w = x.shape
    return x.reshape(batch_size, -1, _c, _h, _w).transpose(1, 2)


class ChannelAttention(nn.Module):
    """
    (N, C, H, W)            (N, C, 1, 1)      (N, C/λ, 1, 1)         (N, C, 1, 1)
     ---> (Spacial Average Pool) ---> Conv2d ---> ReLU ---> Conv2d ---> Sigmoid --->
    |                                                                             |
    x ------------------------------------------------------------------------->(*) ---> output
    """
    def __init__(self, in_channels: int, num_bits: int = 32, extract_coefficient: float = 8):
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


class CBAM(nn.Module):  # Convolution Block Attention Module
    """
    x ---> Channel Attention ---> Spacial Attention ---> output
    """
    def __init__(self, in_channels: int):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels=in_channels)
        self.spacial_attention = SpacialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spacial_attention(self.channel_attention(x))

class ChannelResidualModule(nn.Module):
    """
         ----------------------------------------------------------------------------->
       /                                           ---> Channel Attention --->       /
     /                                           /                          /      /
    x ---> Conv2d ---> LeakyRelu ---> Conv2d ---> -----------------------> * ---> + ---> y
    """
    def __init__(self, in_channels: int, res_scale: float = 0.1):
        super(ChannelResidualModule, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)
        )
        self.conv2 = nn.Conv2d(
            in_channels, in_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)
        )
        self.res_scale = res_scale
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.channel_attention = ChannelAttention(in_channels=in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.conv2(self.leaky_relu(self.conv1(x)))
        res = self.channel_attention(res)
        return res * self.res_scale + x


class SpacialResidualModule(nn.Module):
    """
         ----------------------------------------------------------------------------->
       /                                           ---> Spacial Attention --->       /
     /                                           /                          /      /
    x ---> Conv2d ---> LeakyRelu ---> Conv2d ---> -----------------------> * ---> + ---> y
    """
    def __init__(self, in_channels: int, res_scale: float = 0.1):
        super(SpacialResidualModule, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conv2 = nn.Conv2d(
            in_channels, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.res_scale = res_scale
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.spacial_attention = SpacialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.conv2(self.leaky_relu(self.conv1(x)))
        res = self.spacial_attention(res)
        return res * self.res_scale + x


class ResidualBlock(nn.Module):
    """
     _______
    |_Unit_| : x ---> ChannelResidualModule ---> SpacialResidualModule ---> y

              --------------------------------------------------------->
            /      _______       _______                _______       /
    Block: x ---> |_Unit_| ---> |_Unit_| ---> ... ---> |_Unit_| ---> + ---> y
    """
    def __init__(self, in_channels: int, res_scale: float = 0.1, num_blocks: int = 3):
        super(ResidualBlock, self).__init__()
        self.num_blocks = num_blocks
        self.in_channels = in_channels
        self.res_scale = res_scale
        self.residule_block = self.make_layers()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.residule_block(x) + x

    def make_layers(self):
        layers = []
        for _ in range(self.num_blocks):
            layers.append(ChannelResidualModule(in_channels=self.in_channels, res_scale=self.res_scale))
            layers.append(SpacialResidualModule(in_channels=self.in_channels, res_scale=self.res_scale))
        return nn.Sequential(*layers)


class SuperResolutionBlock(nn.Module):
    """
    x ---> Conv2d ---> ResidualBlock ---> up sample ---> Conv2d ---> y
    """
    def __init__(
            self, in_channels: int, scale: int = 2, num_feas: int = 256, res_scale: float = 0.1,
            num_blocks: int = 3
    ):
        super(SuperResolutionBlock, self).__init__()
        self.fea_conv = nn.Conv2d(
            in_channels, num_feas, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.backbone = ResidualBlock(in_channels=num_feas, res_scale=res_scale, num_blocks=num_blocks)
        self.up_sample = UpSample2d(in_channels=num_feas, scale=scale)
        self.tail_conv = nn.Conv2d(
            num_feas, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(self.fea_conv(x))
        return self.tail_conv(self.up_sample(x))

class Conv3dUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: Optional[int] = None):
        out_channels = in_channels if out_channels is None else out_channels
        super(Conv3dUnit, self).__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(
                in_channels, out_channels, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(
                out_channels, out_channels, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
        return self.conv3d(x)


class FocusUnit(nn.Module):
    def __init__(self, in_channels: int):
        super(FocusUnit, self).__init__()
        self.conv1 = Conv3dUnit(in_channels=in_channels)
        self.conv2 = Conv3dUnit(in_channels=in_channels)
        self.conv3 = Conv3dUnit(in_channels=in_channels)
        self.conv4 = Conv3dUnit(in_channels=in_channels)
        self.channel_attention = ChannelAttention(in_channels=in_channels)
        self.spacial_attention = SpacialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res1 = self.conv2(self.channel_attention(self.conv1(x)))
        x = res1 + x
        return self.conv4(self.spacial_attention(self.conv3(x))) + x


class UpSample2d(nn.Module):
    """
    (N, C, H, W)   (N, C * Scale * Scale, H, W)        (N, C, H * Scale, W * Scale)
    x ---> Conv2d --------------> --------------> PixelShuffle ---> y
    """
    def __init__(self, in_channels: int, scale: int):
        super(UpSample2d, self).__init__()
        self.up_conv = nn.Conv2d(
            in_channels, in_channels * (scale ** 2), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.pixel_shuffle = nn.PixelShuffle(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pixel_shuffle(self.up_conv(x))


class UpSample3d(nn.Module):
    def __init__(self, in_channels: int, scale: int):
        super(UpSample3d, self).__init__()
        self.up_conv = nn.Conv2d(
            in_channels, in_channels * (scale ** 2), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.pixel_shuffle = nn.PixelShuffle(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_4d = to_4d_tensor(x)
        return to_5d_tensor(
            self.pixel_shuffle(self.up_conv(x_4d)), batch_size=x.shape[0]
        )


class FocusBlock(nn.Module):
    def __init__(self, num_feas: int, scale: int, num_units: int = 3):
        super(FocusBlock, self).__init__()
        self.num_feas = num_feas
        self.head = Conv3dUnit(in_channels=1, out_channels=num_feas)
        self.backbone = self.make_layer(num_units)
        self.tail = nn.Sequential(
            UpSample2d(num_feas, scale=scale), Conv3dUnit(num_feas, 1)
        )

    def make_layer(self, num: int):
        layers = []
        for _ in range(num):
            layers.append(FocusUnit(in_channels=self.num_feas))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        x = self.backbone(x) + x
        return self.tail(x)


class FocusRestoreMethod3d(nn.Module):
    def __init__(
            self, in_channels: int, out_channels: int, scale: int, num_groups: int = 8, num_feas: int = 64,
            shared_params: bool = True, branch_units: int = 24
    ):
        super(FocusRestoreMethod3d, self).__init__()
        self.num_groups = num_groups
        self.shared_params = shared_params
        self.num_feas = num_feas
        self.scale = scale
        if shared_params:
            self.branch_1 = FocusBlock(num_feas=num_feas, scale=int(math.sqrt(scale)), num_units=branch_units)
        else:
            self.branch_1_list = []
            for i in range(num_groups):
                self.branch_1_list.append(FocusBlock(
                    num_feas=num_feas, scale=int(math.sqrt(scale)), num_units=branch_units
                ))
        self.branch_2 = FocusBlock(num_feas=num_feas, scale=int(math.sqrt(scale)), num_units=branch_units)
        self.global_skip_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.channel_up_scale = nn.Conv2d(
            in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _n, _d, _h, _w = x.shape
        group_interval = math.floor(_d / self.num_groups)
        y = torch.zeros(
            size=(_n, 1, _d, _h * int(math.sqrt(self.scale)), int(_w * math.sqrt(self.scale)))
        ).to(x.device)
        for i in range(self.num_groups):
            if i == self.num_groups - 1:
                xi = x[:, i * group_interval:, :, :]
                if self.shared_params:
                    y[:, :, i * group_interval:, :, :] = self.branch_1(xi)
                else:
                    y[:, :, i * group_interval:, :, :] = self.branch_1_list[i](xi)
            else:
                xi = x[:, i * group_interval: (i + 1) * group_interval, :, :]
                if self.shared_params:
                    y[:, :, i * group_interval: (i + 1) * group_interval, :, :] = self.branch_1(xi)
                else:
                    y[:, :, i * group_interval: (i + 1) * group_interval, :, :] = self.branch_1_list[i](xi)
        interpolate = self.global_skip_conv(F.interpolate(x, scale_factor=4, mode='bicubic'))
        y = self.branch_2(y).squeeze(1) + interpolate
        return self.channel_up_scale(y)


class F3DCN(nn.Module):
    def __init__(self, num_bits: int = 32):
        super(F3DCN, self).__init__()
        self.encoder = FocusEncoder(172, 27, num_bits = num_bits)
        self.decoder = FocusRestoreMethod3d(in_channels=27, out_channels=172, scale=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


if __name__ == '__main__':
    device = 'cuda:2'
    x = torch.randn(8, 172, 128, 4).to(device)
    # model = FocusCompressMethod(27, 172, scale=4).to(device)
    # model = FocusEncoder(172, 27).to(device)
    model = Focus2dCompressNetwork(num_bits=32).to(device)
    total_param = sum([param.numel() for param in model.decoder.parameters()])
    print(total_param)
    print(model(x).shape)
    print(model(x))
    # print(model.encoder.spacial_attention.conv.weight)
    spacial_attention = SpacialAttention(num_bits=32).to(device)
    print(spacial_attention(x).shape)
    # for name, param in model.named_parameters():
    #     print(name)
    #     print(param.shape)
    state = torch.load('./f2dcn_param/F2DCN_32_bit_12960_epoch.pth')
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
