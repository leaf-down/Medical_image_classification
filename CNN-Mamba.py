import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ConvTConvPW(nn.Module):
    def __init__(self,
                 in_channels,
                 kernel1=3,
                 kernel2=5,
                 kernel3=1,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.k1 = kernel1
        self.k2 = kernel2
        self.k3 = kernel3  # 逐点卷积，一般不做更改

        self.act = nn.ReLU()
        self.bn = nn.BatchNorm2d(in_channels)

        # 第一层卷积，输入输出一致
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=in_channels,
                               kernel_size=self.k1,
                               stride=1,
                               padding=(self.k1 - 1) // 2)

        # 第二层卷积，先转置图像
        self.conv2 = nn.Conv2d(in_channels=in_channels,
                               out_channels=in_channels,
                               kernel_size=self.k2,
                               stride=1,
                               padding=(self.k2 - 1) // 2)

        # 第三层卷积,逐点卷积
        self.PW_conv = nn.Conv2d(in_channels=in_channels,
                                 out_channels=in_channels,
                                 kernel_size=self.k3)

    def forward(self, x):
        identity = x

        x = self.bn(x)
        x = self.conv1(x)

        torch.flip(x, dims=[2, 3])

        x = self.act(self.bn(x))
        x = self.conv2(x)

        torch.flip(x, dims=[2, 3])

        x += identity
        out = self.PW_conv(x)

        return out


class SS2D_with_SSD(nn.Module):
    def __init__(self,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)


