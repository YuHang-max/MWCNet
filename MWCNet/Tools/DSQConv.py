import torch
import torch.nn as nn
import torch.nn.functional as F

device_ids = [2]
device = torch.device(f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu')

class RoundWithGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        delta = torch.max(x) - torch.min(x)
        x = (x / delta + 0.5)
        return x.round() * 2 - 1

    @staticmethod
    def backward(ctx, g):
        return g


class DSQConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 momentum=0.1,
                 num_w_bit=8, num_a_bit=8, QInput=True, bSetQ=True):
        super(DSQConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.num_w_bit = num_w_bit
        self.num_a_bit = num_a_bit
        self.quan_input = QInput
        self.bit_w_range = 2 ** self.num_w_bit - 1
        self.bit_a_range = 2 ** self.num_a_bit - 1
        self.is_quan = bSetQ
        self.momentum = momentum
        
        if self.is_quan:
            # 权重量化参数 - 自动根据设备配置分配
            self.uW = nn.Parameter(data=torch.tensor(2 **31 - 1).float())
            self.lW = nn.Parameter(data=torch.tensor((-1)*(2** 32)).float())
            self.register_buffer('running_uw', torch.tensor([self.uW.data]))
            self.register_buffer('running_lw', torch.tensor([self.lW.data]))
            self.alphaW = nn.Parameter(data=torch.tensor(0.2).float())

            # 偏置量化参数（如果有偏置）
            if self.bias is not None:
                self.uB = nn.Parameter(data=torch.tensor(2 **31 - 1).float())
                self.lB = nn.Parameter(data=torch.tensor((-1)*(2** 32)).float())
                self.register_buffer('running_uB', torch.tensor([self.uB.data]))
                self.register_buffer('running_lB', torch.tensor([self.lB.data]))
                self.alphaB = nn.Parameter(data=torch.tensor(0.2).float())

            # 输入激活量化参数
            if self.quan_input:
                self.uA = nn.Parameter(data=torch.tensor(2 **31 - 1).float())
                self.lA = nn.Parameter(data=torch.tensor((-1)*(2** 32)).float())
                self.register_buffer('running_uA', torch.tensor([self.uA.data]))
                self.register_buffer('running_lA', torch.tensor([self.lA.data]))
                self.alphaA = nn.Parameter(data=torch.tensor(0.2).float())

        # 确保所有参数和缓冲区移动到正确的设备
        self.to(device)

    def clipping(self, x, upper, lower):
        x = x + F.relu(lower - x)
        x = x - F.relu(x - upper)
        return x

    def phi_function(self, x, mi, alpha, delta):
        alpha = torch.where(alpha >= 2.0, torch.tensor([2.0], device=device), alpha)
        s = 1 / (1 - alpha)
        k = (torch.log(2 / alpha - 1)) * (1 / delta)
        x = (torch.tanh((x - mi) * k)) * s
        return x

    def sgn(self, x):
        return RoundWithGradient.apply(x)

    def dequantize(self, x, lower_bound, delta, interval):
        return ((x + 1) / 2 + interval) * delta + lower_bound

    def forward(self, x):
        if self.is_quan:
            # 权重量化
            if self.training:
                cur_running_lw = self.running_lw.mul(1 - self.momentum).add(self.momentum * self.lW)
                cur_running_uw = self.running_uw.mul(1 - self.momentum).add(self.momentum * self.uW)
            else:
                cur_running_lw = self.running_lw
                cur_running_uw = self.running_uw

            Qweight = self.clipping(self.weight, cur_running_uw, cur_running_lw)
            cur_max = torch.max(Qweight)
            cur_min = torch.min(Qweight)
            delta = (cur_max - cur_min) / self.bit_w_range
            interval = ((Qweight - cur_min) / delta).trunc()
            mi = (interval + 0.5) * delta + cur_min
            Qweight = self.phi_function(Qweight, mi, self.alphaW, delta)
            Qweight = self.sgn(Qweight)
            Qweight = self.dequantize(Qweight, cur_min, delta, interval)

            # 偏置量化（如果有）
            Qbias = self.bias
            if self.bias is not None:
                if self.training:
                    cur_running_lB = self.running_lB.mul(1 - self.momentum).add(self.momentum * self.lB)
                    cur_running_uB = self.running_uB.mul(1 - self.momentum).add(self.momentum * self.uB)
                else:
                    cur_running_lB = self.running_lB
                    cur_running_uB = self.running_uB

                Qbias = self.clipping(self.bias, cur_running_uB, cur_running_lB)
                cur_max = torch.max(Qbias)
                cur_min = torch.min(Qbias)
                delta = (cur_max - cur_min) / self.bit_w_range
                interval = ((Qbias - cur_min) / delta).trunc()
                mi = (interval + 0.5) * delta + cur_min
                Qbias = self.phi_function(Qbias, mi, self.alphaB, delta)
                Qbias = self.sgn(Qbias)
                Qbias = self.dequantize(Qbias, cur_min, delta, interval)

            # 输入激活量化
            Qactivation = x
            if self.quan_input:
                if self.training:
                    cur_running_lA = self.running_lA.mul(1 - self.momentum).add(self.momentum * self.lA)
                    cur_running_uA = self.running_uA.mul(1 - self.momentum).add(self.momentum * self.uA)
                else:
                    cur_running_lA = self.running_lA
                    cur_running_uA = self.running_uA

                Qactivation = self.clipping(x, cur_running_uA, cur_running_lA)
                cur_max = torch.max(Qactivation)
                cur_min = torch.min(Qactivation)
                delta = (cur_max - cur_min) / self.bit_a_range
                interval = ((Qactivation - cur_min) / delta).trunc()
                mi = (interval + 0.5) * delta + cur_min
                Qactivation = self.phi_function(Qactivation, mi, self.alphaA, delta)
                Qactivation = self.sgn(Qactivation)
                Qactivation = self.dequantize(Qactivation, cur_min, delta, interval)

            # 卷积操作（修正重复卷积问题）
            output = F.conv2d(
                Qactivation, Qweight, Qbias, 
                self.stride, self.padding, self.dilation, self.groups
            )

        else:
            output = F.conv2d(
                x, self.weight, self.bias, 
                self.stride, self.padding, self.dilation, self.groups
            )

        return output


class DSQAct(nn.Module):
    def __init__(self, num_a_bit=8, momentum=0.1, quan_input=True):
        super(DSQAct, self).__init__()
        self.num_a_bit = num_a_bit
        self.bit_a_range = 2 ** self.num_a_bit - 1
        self.momentum = momentum
        self.quan_input = quan_input

        if self.quan_input:
            self.uA = nn.Parameter(data=torch.tensor(2 **31 - 1).float())
            self.lA = nn.Parameter(data=torch.tensor((-1)*(2** 32)).float())
            self.register_buffer('running_uA', torch.tensor([self.uA.data]))
            self.register_buffer('running_lA', torch.tensor([self.lA.data]))
            self.alphaA = nn.Parameter(data=torch.tensor(0.2).float())

        # 确保所有参数和缓冲区移动到正确的设备
        self.to(device)

    def clipping(self, x, upper, lower):
        x = x + F.relu(lower - x)
        x = x - F.relu(x - upper)
        return x

    def phi_function(self, x, mi, alpha, delta):
        alpha = torch.where(alpha >= 2.0, torch.tensor([2.0], device=device), alpha)
        s = 1 / (1 - alpha)
        k = (torch.log(2 / alpha - 1)) * (1 / delta)
        x = (torch.tanh((x - mi) * k)) * s
        return x

    def sgn(self, x):
        return RoundWithGradient.apply(x)

    def dequantize(self, x, lower_bound, delta, interval):
        return ((x + 1) / 2 + interval) * delta + lower_bound

    def forward(self, x):
        if self.quan_input:
            if self.training:
                cur_running_lA = self.running_lA.mul(1 - self.momentum).add(self.momentum * self.lA)
                cur_running_uA = self.running_uA.mul(1 - self.momentum).add(self.momentum * self.uA)
            else:
                cur_running_lA = self.running_lA
                cur_running_uA = self.running_uA

            Qactivation = self.clipping(x, cur_running_uA, cur_running_lA)
            cur_max = torch.max(Qactivation)
            cur_min = torch.min(Qactivation)
            delta = (cur_max - cur_min) / self.bit_a_range
            interval = ((Qactivation - cur_min) / delta).trunc()
            mi = (interval + 0.5) * delta + cur_min
            Qactivation = self.phi_function(Qactivation, mi, self.alphaA, delta)
            Qactivation = self.sgn(Qactivation)
            output = Qactivation
        else:
            output = x
        return output
