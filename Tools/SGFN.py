import torch
import torch.nn as nn
from .DIMChange import to_4d,to_3d
class SpatialGate(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim) # DW Conv

    def forward(self, x, H, W):
        # Split
        x1, x2 = x.chunk(2, dim = -1)
        B, N, C = x.shape
        x2 = self.conv(self.norm(x2).transpose(1, 2).contiguous().view(B, C//2, H, W)).flatten(2).transpose(-1, -2).contiguous()

        return x1 * x2

class SGFN(nn.Module):
    """ Spatial-Gate Feed-Forward Network.
    Args:
        in_features (int): Number of input channels.
        hidden_features (int | None): Number of hidden channels. Default: None
        out_features (int | None): Number of output channels. Default: None
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        drop (float): Dropout rate. Default: 0.0
    """
    def __init__(self, in_features, hidden_features=None, out_features = None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = in_features
        hidden_features = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.sg = SpatialGate(hidden_features//2)
        self.fc2 = nn.Linear(hidden_features//2, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        Input: x: (B, H*W, C), H, W
        Output: x: (B, H*W, C)
        """
        b, c, h, w = x.shape
        x = to_3d(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.sg(x, h, w)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)

        x = x.transpose(1, 2).reshape(-1, c, h, w)
        return x


if __name__ == '__main__':
    # 定义输入参数
    batch_size = 1
    height = 128  # 假设图像高度为32
    width = 4  # 假设图像宽度为32
    channels = 172  # 输入通道数

    block = SGFN(in_features=channels)

    # 创建随机输入数据 (B, H*W, C)
    x = torch.randn(batch_size, height * width, channels)

    # 前向传播并打印输入输出的形状
    output = block(x, height, width)

    print(x.size())
    print(output.size())