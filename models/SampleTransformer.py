import torch
from torch import nn
from torch.nn import functional as F
from typing import Union, List, Tuple, Optional


def sigmoid_quantile(
        x: torch.Tensor, medium: torch.Tensor, alpha: Union[torch.Tensor, float],
        delta: torch.Tensor, interval: torch.Tensor, min_: torch.Tensor
) -> torch.Tensor:
    # lamda: λ = e^(-α·δ/2) + 1, to smooth the curve transition
    lamda = torch.exp((-1) * alpha * delta / 2) + 1

    # activation: params under sigmoid activation, result falls in [0, 1]
    activation = lamda / (1 + torch.exp((-1) * alpha * (x - medium)))

    # set activation to either 0 or 1
    activation = (torch.round(activation) - activation).detach() + activation

    # return quantile result
    return (interval + activation) * delta + min_


def quantile_params(
        params: torch.Tensor, upper_bound: torch.Tensor, lower_bound: torch.Tensor,
        quantile_range: int, alpha: Union[torch.Tensor, float]
) -> torch.Tensor:
    # clip the param within (lower_bound, upper_bound)
    params = torch.clip(params, lower_bound, upper_bound)

    if params.numel() == 1:
        return params

    # delta: the length of every interval
    max_, min_ = torch.max(params), torch.min(params)
    delta = (max_ - min_) / quantile_range

    # interval: which interval every single number falls in
    interval = ((params - min_) / delta).trunc()

    # medium: medium position of every interval
    medium = (interval + 0.5) * delta + min_

    # return quantile params
    return sigmoid_quantile(params, medium, alpha, delta, interval, min_)


def init_bounds() -> Tuple[
    nn.Parameter, nn.Parameter, nn.Parameter, nn.Parameter, nn.Parameter, nn.Parameter
]:
    # uw: upper weight, set upper bound of weight
    # lw: lower weight, set lower bound of weight
    uw = nn.Parameter(torch.tensor(2 ** 31 - 1, dtype=torch.float32))
    lw = nn.Parameter(torch.tensor(-2 ** 31, dtype=torch.float32))

    # ub: upper bias, set upper bound of bias
    # lb: lower bias, set lower bound of bias
    ub = nn.Parameter(torch.tensor(2 ** 31 - 1, dtype=torch.float32))
    lb = nn.Parameter(torch.tensor(-2 ** 31, dtype=torch.float32))

    # ux: upper x, set upper bound of input tensor
    # lx: lower x, set lower bound of input tensor
    ux = nn.Parameter(torch.tensor(2 ** 31 - 1, dtype=torch.float32))
    lx = nn.Parameter(torch.tensor(-2 ** 31, dtype=torch.float32))

    return uw, lw, ub, lb, ux, lx


def get_running_bounds(
        uw: Union[torch.Tensor, None] = None, lw: Union[torch.Tensor, None] = None,
        ub: Union[torch.Tensor, None] = None, lb: Union[torch.Tensor, None] = None,
        ux: Union[torch.Tensor, None] = None, lx: Union[torch.Tensor, None] = None,
        training: bool = True, momentum: float = 0.1,
        running_uw: Union[torch.Tensor, None] = None, running_lw: Union[torch.Tensor, None] = None,
        running_ub: Union[torch.Tensor, None] = None, running_lb: Union[torch.Tensor, None] = None,
        running_ux: Union[torch.Tensor, None] = None, running_lx: Union[torch.Tensor, None] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, ]:
    use_running_uw, use_running_lw, use_running_ub, use_running_lb = None, None, None, None
    # on train process, smoothly update params
    if training:
        if uw is not None:
            # use_running_uw and use_running_lw: bounds used in quantile process
            use_running_uw = running_uw * (1 - momentum) + uw * momentum
            use_running_lw = running_lw * (1 - momentum) + lw * momentum

        if ub is not None:
            # use_running_ub and use_running_lb: bounds used in quantile process
            use_running_ub = running_ub * (1 - momentum) + ub * momentum
            use_running_lb = running_lb * (1 - momentum) + lb * momentum

        # use_running_ux and use_running_lx: bounds used in quantile process
        use_running_ux = running_ux * (1 - momentum) + ux * momentum
        use_running_lx = running_lx * (1 - momentum) + lx * momentum

    # on valid process, fix the bounds
    else:
        if uw is not  None:
            # use_running_uw and use_running_lw: fixed bound
            use_running_uw = running_uw
            use_running_lw = running_lw

        if ub is not None:
            # use_running_ub and use_running_lb: fixed bound
            use_running_ub = running_ub
            use_running_lb = running_lb

        # use_running_ux and use_running_lx: fixed bound
        use_running_ux = running_ux
        use_running_lx = running_lx

    return (use_running_uw, use_running_lw, use_running_ub, use_running_lb,
            use_running_ux, use_running_lx)


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.xavier_normal_(m.bias.data.unsqueeze(0))
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.xavier_normal_(m.bias.data.unsqueeze(0))
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class QuantileActivate(nn.Module):
    def __init__(self, use_quantile: bool = True, momentum: float = 0.1, num_bits: int = 8, alpha: float = 1.0):
        super(QuantileActivate, self).__init__()
        self.use_quantile = use_quantile
        self.momentum = momentum
        self.quantile_range = 2 ** num_bits - 1
        self.alpha = nn.Parameter(torch.tensor(alpha))
        if use_quantile:
            _, _, _, _, self.ux, self.lx = init_bounds()

            # running_ux or running_lx: fixed param to limit updating speed
            self.register_buffer(name='running_ux', tensor=self.ux)
            self.register_buffer(name='running_lx', tensor=self.lx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # if not use quantile, return original value
        if not self.use_quantile:
            return x

        _, _, _, _, use_running_ux, use_running_lx = get_running_bounds(
            ux=self.ux, lx=self.lx, training=self.training, momentum=self.momentum,
            running_ux=self.running_ux, running_lx=self.running_lx
        )

        # set weight input x to quantile values
        quantile_input = quantile_params(
            x, use_running_ux, use_running_lx, self.quantile_range, self.alpha
        )
        return quantile_input


class QuantileLinear(nn.Linear):
    def __init__(
            self, in_features: int, out_features: int, use_quantile: bool = True,
            momentum: float = 0.1, num_bits: int = 8, alpha: float = 1.0
    ):
        super(QuantileLinear, self).__init__(in_features, out_features)
        self.use_quantile = use_quantile
        self.momentum = momentum
        self.quantile_range = 2 ** num_bits - 1
        self.alpha = nn.Parameter(torch.tensor(alpha))
        if use_quantile:
            self.uw, self.lw, self.ub, self.lb, self.ux, self.lx = init_bounds()

            # running_uw or running_lw: fixed param to limit updating speed
            self.register_buffer(name='running_uw', tensor=self.uw)
            self.register_buffer(name='running_lw', tensor=self.lw)

            # running_ub or running_lb: fixed param to limit updating speed
            self.register_buffer(name='running_ub', tensor=self.ub)
            self.register_buffer(name='running_lb', tensor=self.lb)

            # running_ux or running_lx: fixed param to limit updating speed
            self.register_buffer(name='running_ux', tensor=self.ux)
            self.register_buffer(name='running_lx', tensor=self.lx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # if not use quantile, return original forward function
        if not self.use_quantile:
            return super().forward(x)

        (use_running_uw, use_running_lw, use_running_ub, use_running_lb,
         use_running_ux, use_running_lx) = get_running_bounds(
            self.uw, self.lw, self.ub, self.lb, self.ux, self.lx, self.training, self.momentum,
            self.running_uw, self.running_lw, self.running_ub, self.running_lb,
            self.running_ux, self.running_lx
        )

        # set weight, bias and input x to quantile values
        quantile_weight = quantile_params(
            self.weight, use_running_uw, use_running_lw, self.quantile_range, self.alpha
        )
        quantile_bias = quantile_params(
            self.bias, use_running_ub, use_running_lb, self.quantile_range, self.alpha
        )
        quantile_input = quantile_params(
            x, use_running_ux, use_running_lx, self.quantile_range, self.alpha
        )
        return torch.matmul(
            quantile_input, quantile_weight.transpose(0, 1)
        ) + quantile_bias


class QuantileConv2d(nn.Conv2d):
    def __init__(
            self, in_channels: int, out_channels: int,
            kernel_size: Union[int, List[int], Tuple[int, int]],
            stride: Union[int, List[int], Tuple[int, int]],
            padding: Union[int, List[int], Tuple[int, int]],
            use_quantile: bool = True, momentum: float = 0.1, num_bits: int = 8,
            alpha: float = 1.0,
    ):
        super(QuantileConv2d, self).__init__(in_channels, out_channels, kernel_size)
        self.use_quantile = use_quantile
        self.momentum = momentum
        self.quantile_range = 2 ** num_bits - 1
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.stride = stride
        self.padding = padding
        if use_quantile:
            self.uw, self.lw, self.ub, self.lb, self.ux, self.lx = init_bounds()

            # running_uw or running_lw: fixed param to limit updating speed
            self.register_buffer(name='running_uw', tensor=self.uw)
            self.register_buffer(name='running_lw', tensor=self.lw)

            # running_ub or running_lb: fixed param to limit updating speed
            self.register_buffer(name='running_ub', tensor=self.ub)
            self.register_buffer(name='running_lb', tensor=self.lb)

            # running_ux or running_lx: fixed param to limit updating speed
            self.register_buffer(name='running_ux', tensor=self.ux)
            self.register_buffer(name='running_lx', tensor=self.lx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_quantile:
            return super().forward(x)

        (use_running_uw, use_running_lw, use_running_ub, use_running_lb,
         use_running_ux, use_running_lx) = get_running_bounds(
            self.uw, self.lw, self.ub, self.lb, self.ux, self.lx, self.training, self.momentum,
            self.running_uw, self.running_lw, self.running_ub, self.running_lb,
            self.running_ux, self.running_lx
        )

        # set weight, bias and input x to quantile values
        quantile_weight = quantile_params(
            self.weight, use_running_uw, use_running_lw, self.quantile_range, self.alpha
        )
        quantile_bias = quantile_params(
            self.bias, use_running_ub, use_running_lb, self.quantile_range, self.alpha
        )
        quantile_input = quantile_params(
            x, use_running_ux, use_running_lx, self.quantile_range, self.alpha
        )
        return F.conv2d(
            quantile_input, quantile_weight, quantile_bias, self.stride, self.padding
        )


class QuantileConvTranspose2d(nn.ConvTranspose2d):
    def __init__(
            self, in_channels: int, out_channels: int,
            kernel_size: Union[int, List[int], Tuple[int, int]],
            stride: Union[int, List[int], Tuple[int, int]],
            padding: Union[int, List[int], Tuple[int, int]],
            output_padding: Union[int, List[int], Tuple[int, int]],
            use_quantile: bool = True, momentum: float = 0.1, num_bits: int = 8,
            alpha: float = 1.0,
    ):
        super(QuantileConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size)
        self.use_quantile = use_quantile
        self.momentum = momentum
        self.quantile_range = 2 ** num_bits - 1
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        if use_quantile:
            self.uw, self.lw, self.ub, self.lb, self.ux, self.lx = init_bounds()

            # running_uw or running_lw: fixed param to limit updating speed
            self.register_buffer(name='running_uw', tensor=self.uw)
            self.register_buffer(name='running_lw', tensor=self.lw)

            # running_ub or running_lb: fixed param to limit updating speed
            self.register_buffer(name='running_ub', tensor=self.ub)
            self.register_buffer(name='running_lb', tensor=self.lb)

            # running_ux or running_lx: fixed param to limit updating speed
            self.register_buffer(name='running_ux', tensor=self.ux)
            self.register_buffer(name='running_lx', tensor=self.lx)

    def forward(
            self, x: torch.Tensor, output_size: Optional[List[int]] = None
    ) -> torch.Tensor:
        if not self.use_quantile:
            return super().forward(x)

        (use_running_uw, use_running_lw, use_running_ub, use_running_lb,
         use_running_ux, use_running_lx) = get_running_bounds(
            self.uw, self.lw, self.ub, self.lb, self.ux, self.lx, self.training, self.momentum,
            self.running_uw, self.running_lw, self.running_ub, self.running_lb,
            self.running_ux, self.running_lx
        )

        # set weight, bias and input x to quantile values
        quantile_weight = quantile_params(
            self.weight, use_running_uw, use_running_lw, self.quantile_range, self.alpha
        )
        quantile_bias = quantile_params(
            self.bias, use_running_ub, use_running_lb, self.quantile_range, self.alpha
        )
        quantile_input = quantile_params(
            x, use_running_ux, use_running_lx, self.quantile_range, self.alpha
        )
        return F.conv_transpose2d(
            quantile_input, quantile_weight, quantile_bias, self.stride, self.padding,
            self.output_padding
        )


class SampleAttention(nn.Module):
    def __init__(
            self, input_channels: int, output_channels: int, use_quantile: bool = False,
            h_scale: int = 1, w_scale: int = 1, mode: str = 'downsample', num_bits: int = 8
    ):
        super(SampleAttention, self).__init__()
        self.out_channels = output_channels
        self.h_scale = h_scale
        self.w_scale = w_scale

        self.linearQ = QuantileLinear(input_channels, output_channels, use_quantile=use_quantile, num_bits=num_bits)
        self.linearK = QuantileLinear(input_channels, output_channels, use_quantile=use_quantile, num_bits=num_bits)
        self.linearV = QuantileLinear(input_channels, output_channels, use_quantile=use_quantile, num_bits=num_bits)
        self.linearLast = QuantileLinear(output_channels, output_channels, use_quantile=use_quantile, num_bits=num_bits)

        self.mode = mode
        if mode == 'downsample':
            self.sample = QuantileConv2d(
                in_channels=input_channels, out_channels=input_channels,
                kernel_size=(3, 3), stride=(h_scale, w_scale), padding=(1, 1),
                use_quantile=use_quantile, num_bits=num_bits
            )
        elif mode == 'upsample':
            self.sample = QuantileConvTranspose2d(
                in_channels=input_channels, out_channels=input_channels,
                kernel_size=(5, 5), stride=(h_scale, w_scale), padding=(2, 2),
                output_padding=(h_scale - 1, w_scale - 1),
                use_quantile=use_quantile, num_bits=num_bits
            )
        else:
            raise ValueError(
                f'Input mode: {mode} incorrect! Expected mode in (\'downsample\', \'upsample\').'
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, channels, h, w)
        batch_size, channels, h, w = x.shape

        # downsample: (batch_size, h // h_scale, w // w_scale, channels)
        # upsample: (batch_size, h * h_scale, w * w_scale, channels)
        sampled = self.sample(x).permute(0, 2, 3, 1)

        # downsample: (batch_size, h // h_scale * w // w_scale, channels)
        # upsample: (batch_size, h * h_scale * w * w_scale, channels)
        q = self.linearQ(sampled).reshape(batch_size, -1, self.out_channels)

        # (batch_size, h, w, channels)
        x = x.permute(0, 2, 3, 1)

        # (batch_size, channels, h * w)
        k = self.linearK(x).permute(0, 3, 1, 2).reshape(batch_size, self.out_channels, -1)

        # (batch_size, h * w, channels)
        v = self.linearV(x).reshape(batch_size, -1, self.out_channels)

        # downsample: (batch_size, h // h_scale * w // w_scale, h * w)
        # upsample: (batch_size, h * h_scale * w * w_scale, h * w)
        q_mat_k = torch.matmul(q, k)
        activation = F.softmax(q_mat_k, dim=-1)

        # downsample: (batch_size, h // h_scale, w // w_scale, channels)
        # upsample: (batch_size, h * h_scale, w * w_scale, channels)
        if self.mode == 'downsample':
            act_mat_v = (torch.matmul(activation, v)
                         .reshape(batch_size, h // self.h_scale, w // self.w_scale, -1))
        else:
            act_mat_v = (torch.matmul(activation, v)
                         .reshape(batch_size, h * self.h_scale, w * self.w_scale, -1))

        # downsample: (batch_size, channels, h // h_scale * w // w_scale)
        # upsample: (batch_size, channels, h * h_scale * w * w_scale)
        res = self.linearLast(act_mat_v).permute(0, 3, 1, 2)

        return res


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, use_quantile: bool = False, num_bits: int = 8):
        super(ConvBlock, self).__init__()
        self.conv1 = QuantileConv2d(
            in_channels=input_channels, out_channels=input_channels * 2,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
            use_quantile=use_quantile, num_bits=num_bits
        )
        self.conv2 = QuantileConv2d(
            in_channels=input_channels * 2, out_channels=input_channels * 2,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
            use_quantile=use_quantile, num_bits=num_bits
        )
        self.conv3 = QuantileConv2d(
            in_channels=input_channels * 2, out_channels=output_channels,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
            use_quantile=use_quantile, num_bits=num_bits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        return self.conv3(x)


# Downsample or Upsample Attention Block
class SABlock(nn.Module):
    def __init__(
            self, input_channels: int, output_channels: int,
            h_scale: int = 1, w_scale: int = 1, mode: str = 'downsample',
            use_quantile: bool = False, num_bits: int = 8
    ):
        super(SABlock, self).__init__()
        self.mode = mode
        self.out_channels = output_channels
        self.h_scale, self.w_scale = h_scale, w_scale

        self.attention = SampleAttention(
            input_channels=input_channels, output_channels=output_channels,
            h_scale=h_scale, w_scale=w_scale, mode=mode, use_quantile=use_quantile, num_bits=num_bits
        )
        self.conv = ConvBlock(
            input_channels=output_channels, output_channels=output_channels,
            use_quantile=use_quantile, num_bits=num_bits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, channels, h, w)
        batch_size, channels, h, w = x.shape

        if self.mode == 'downsample':
            norm = nn.LayerNorm(
                normalized_shape=[self.out_channels, h // self.h_scale, w // self.w_scale]
            ).to(x.device)
        elif self.mode == 'upsample':
            norm = nn.LayerNorm(
                normalized_shape=[self.out_channels, h * self.h_scale, w * self.w_scale]
            ).to(x.device)
        else:
            raise ValueError(
                f'Input mode: {self.mode} incorrect! Expected mode in (\'downsample\', \'upsample\').'
            )

        # downsample: (batch_size, out_channels, h // h_scale, w // w_scale)
        # upsample: (batch_size, out_channels, h * h_scale, w * w_scale)
        if self.mode == 'downsample':
            interpolated = F.interpolate(
                x.reshape(batch_size, channels, -1).unsqueeze(1),
                size=[self.out_channels, h // self.h_scale * w // self.w_scale],
                mode='bilinear', align_corners=False
            ).squeeze(1).reshape(
                batch_size, self.out_channels, h // self.h_scale, w // self.w_scale
            )
        else:
            interpolated = F.interpolate(
                x.reshape(batch_size, channels, -1).unsqueeze(1),
                size=[self.out_channels, h * self.h_scale * w * self.w_scale],
                mode='bilinear', align_corners=False
            ).squeeze(1).reshape(
                batch_size, self.out_channels, h * self.h_scale, w * self.w_scale
            )

        x = interpolated + norm(self.attention(x))
        return x + norm(self.conv(x))


class Encoder(nn.Module):
    def __init__(self, use_quantile: bool = True, num_bits: int = 8):
        super(Encoder, self).__init__()
        self.DSA1 = SABlock(
            172, 128, 2, 1, mode='downsample',
            use_quantile=use_quantile, num_bits=num_bits
        )
        self.DSA2 = SABlock(
            128, 64, 1, 2, mode='downsample',
            use_quantile=use_quantile, num_bits=num_bits
        )
        self.DSA3 = SABlock(
            64, 48, 2, 1, mode='downsample',
            use_quantile=use_quantile, num_bits=num_bits
        )
        self.DSA4 = SABlock(
            48, 27, 1, 2, mode='downsample',
            use_quantile=use_quantile, num_bits=num_bits
        )

        self.conv = ConvBlock(27, 27, use_quantile=use_quantile, num_bits=num_bits)
        self.activate = QuantileActivate(use_quantile=use_quantile, num_bits=num_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, h, w = x.shape
        interploated = F.interpolate(
            x.reshape(batch_size, channels, h * w).unsqueeze(1),
            size=(27, (h // 4) * (w // 4)), mode='bilinear', align_corners=False
        ).squeeze(1).reshape(batch_size, 27, (h // 4), (w // 4))
        x = self.DSA1(x)
        x = self.DSA2(x)
        x = self.DSA3(x)
        x = self.DSA4(x)

        return self.activate(interploated + self.conv(x))


class LightEncoder(nn.Module):
    def __init__(self, use_quantile: bool = True, num_bits: int = 8):
        super(LightEncoder, self).__init__()
        self.DSA1 = SABlock(
            172, 96, 2, 2, mode='downsample',
            use_quantile=use_quantile, num_bits=num_bits
        )
        self.DSA2 = SABlock(
            96, 32, 2, 2, mode='downsample',
            use_quantile=use_quantile, num_bits=num_bits
        )

        self.conv = ConvBlock(32, 27, use_quantile=use_quantile, num_bits=num_bits)
        self.activate = QuantileActivate(use_quantile=use_quantile, num_bits=num_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, h, w = x.shape
        interploated = F.interpolate(
            x.reshape(batch_size, channels, h * w).unsqueeze(1),
            size=(27, (h // 4) * (w // 4)), mode='bilinear', align_corners=False
        ).squeeze(1).reshape(batch_size, 27, (h // 4), (w // 4))
        x = self.DSA1(x)
        x = self.DSA2(x)

        return self.activate(interploated + self.conv(x))


class Decoder(nn.Module):
    def __init__(self, use_quantile: bool = False):
        super(Decoder, self).__init__()
        self.USA1 = SABlock(
            27, 54, 1, 1, mode='upsample',
            use_quantile=use_quantile
        )
        self.USA2 = SABlock(
            54, 96, 2, 1, mode='upsample',
            use_quantile=use_quantile
        )
        self.USA3 = SABlock(
            96, 96, 1, 2, mode='upsample',
            use_quantile=use_quantile
        )
        self.USA4 = SABlock(
            96, 128, 2, 1, mode='upsample',
            use_quantile=use_quantile
        )
        self.USA5 = SABlock(
            128, 172, 1, 2, mode='upsample',
            use_quantile=use_quantile
        )

        self.conv = ConvBlock(input_channels=172, output_channels=172, use_quantile=use_quantile)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, h, w = x.shape
        interploated = F.interpolate(
            x.reshape(batch_size, channels, h * w).unsqueeze(1),
            size=(172, (h * 4) * (w * 4)), mode='bilinear', align_corners=False
        ).squeeze(1).reshape(batch_size, 172, (h * 4), (w * 4))
        x = self.USA1(x)
        x = self.USA2(x)
        x = self.USA3(x)
        x = self.USA4(x)
        x = self.USA5(x)

        return interploated + self.conv(x)


class DenseDecoder(nn.Module):
    def __init__(self, use_quantile: bool = False):
        super(DenseDecoder, self).__init__()
        self.USA1 = SABlock(
            27, 200, 1, 1, mode='upsample',
            use_quantile=use_quantile
        )
        self.USA2 = SABlock(
            200, 172, 2, 2, mode='upsample',
            use_quantile=use_quantile
        )
        self.USA3 = SABlock(
            172, 172, 2, 2, mode='upsample',
            use_quantile=use_quantile
        )

        self.conv = ConvBlock(input_channels=172, output_channels=172, use_quantile=use_quantile)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, h, w = x.shape
        interploated = F.interpolate(
            x.reshape(batch_size, channels, h * w).unsqueeze(1),
            size=(172, (h * 4) * (w * 4)), mode='bilinear', align_corners=False
        ).squeeze(1).reshape(batch_size, 172, (h * 4), (w * 4))
        x = self.USA1(x)
        x = self.USA2(x)
        x = self.USA3(x)

        return interploated + self.conv(x)


class SampleTransformer(nn.Module):
    def __init__(self, num_bits: int = 8, snr: float = 0.0):
        super(SampleTransformer, self).__init__()
        self.encoder = Encoder(use_quantile=True, num_bits=num_bits)
        self.decoder = Decoder(use_quantile=False)
        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)
        self.snr = snr

    def awgn(self, x: torch.Tensor, snr: float = 0.0) -> torch.Tensor:
        snr = 10**(snr/10.0)
        xpower = torch.sum(x**2)/x.numel()
        npower = torch.sqrt(xpower / snr)
        return x + torch.randn(x.shape, device=x.device) * npower

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        if self.training:
            x = self.awgn(x, self.snr)
        return self.decoder(x)


class ConvSampleTransformer(nn.Module):
    def __init__(self, num_bits: int = 8, snr: float = 0.0):
        super(ConvSampleTransformer, self).__init__()
        self.encoder = nn.Sequential(
            QuantileConv2d(
                in_channels=172, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                use_quantile=True, num_bits=num_bits
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            QuantileConv2d(
                in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1),
                use_quantile=True, num_bits=num_bits
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            QuantileConv2d(
                in_channels=64, out_channels=27, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1),
                use_quantile=True, num_bits=num_bits
            )
        )
        self.decoder = DenseDecoder(use_quantile=False)
        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)
        self.snr = snr


    def awgn(self, x: torch.Tensor, snr: float = 0.0) -> torch.Tensor:
        snr = 10 ** (snr / 10.0)
        xpower = torch.sum(x ** 2) / x.numel()
        npower = torch.sqrt(xpower / snr)
        return x + torch.randn(x.shape, device=x.device) * npower

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        if self.training:
            x = self.awgn(x, self.snr)
        return self.decoder(x)



class LightSampleTransformer(nn.Module):
    def __init__(self, num_bits: int = 8, snr: float = 0.0):
        super(LightSampleTransformer, self).__init__()
        self.encoder = LightEncoder(use_quantile=True, num_bits=num_bits)
        self.decoder = DenseDecoder(use_quantile=False)
        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)
        self.snr = snr

    def awgn(self, x: torch.Tensor, snr: float = 0.0) -> torch.Tensor:
        snr = 10 ** (snr / 10.0)
        xpower = torch.sum(x ** 2) / x.numel()
        npower = torch.sqrt(xpower / snr)
        return x + torch.randn(x.shape, device=x.device) * npower

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        if self.training:
            x = self.awgn(x, self.snr)
        return self.decoder(x)


class NoquantileSampleTransformer(nn.Module):
    def __init__(self, num_bits: int = 8, snr: float = 0.0):
        super(NoquantileSampleTransformer, self).__init__()
        self.encoder = LightEncoder(use_quantile=False, num_bits=num_bits)
        self.decoder = Decoder(use_quantile=False)
        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)
        self.snr = snr

    def awgn(self, x: torch.Tensor, snr: float = 0.0) -> torch.Tensor:
        snr = 10 ** (snr / 10.0)
        xpower = torch.sum(x ** 2) / x.numel()
        npower = torch.sqrt(xpower / snr)
        return x + torch.randn(x.shape, device=x.device) * npower

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        if self.training:
            x = self.awgn(x, self.snr)
        return self.decoder(x)


if __name__ == '__main__':
    x = torch.randn((3, 172, 128, 4), dtype=torch.float32)
    # y = torch.randn((3, 27, 32, 1), dtype=torch.float32)
    # # print(torch.randn(172 * 128 * 4, 172 * 128 * 4).shape)
    # model = SampleTransformer()

    # # print(model(x).shape)
    # # model = SABlock(
    # #     172, 200, 2, 2, mode='upsample'
    # # )
    #
    # print(model(x).shape)
    model = LightSampleTransformer()
    # for name, param in model.named_parameters():
    #     print(name, param)
    print(model(x).shape)
    print(model(x))
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)  # 6560701
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = torch.sum(model(x))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # print(model.weight)
