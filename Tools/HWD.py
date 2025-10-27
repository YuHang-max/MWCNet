import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward

class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        # 初始化离散小波变换，J=1表示变换的层数，mode='zero'表示填充模式，使用'Haar'小波
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        # 定义卷积、批归一化和ReLU激活的顺序组合
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),  # 1x1卷积层，通道数由in_ch*4变为out_ch
            nn.BatchNorm2d(out_ch),  # 批归一化层
            nn.ReLU(inplace=True),  # ReLU激活函数
        )

    def forward(self, x):
        # 对输入x进行离散小波变换，得到低频部分yL和高频部分yH
        yL, yH = self.wt(x)
        # 提取高频部分的不同分量
        y_HL = yH[0][:, :, 0, ::]  # 水平高频
        y_LH = yH[0][:, :, 1, ::]  # 垂直高频
        y_HH = yH[0][:, :, 2, ::]  # 对角高频
        # 将低频部分和高频部分拼接在一起
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        # 通过卷积、批归一化和ReLU激活处理拼接后的特征
        x = self.conv_bn_relu(x)
        return x