#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import numpy as np

#损失函数SSIM的Pytorch实现
#结构相似性指数（SSIM）用于度量两幅图像之间的结构相似性。和被广泛采用的L2loss不同，SSIM和HVS类似，对局部结构变化的感知敏感.SSIM分为三个部分：照明度、对比度、结构。SSIM值越大代表图像越相似，当两幅图片完全一致的时候SSIM为1，
#loss=1-ssim，由于Pytorch实现了自动求导机制，因此我们只需要实现SSIM loss的前向计算部分即可。不用考虑求导

#计算一维的高斯分布向量
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

#创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
#可以设定channel参数扩展为3通道
def create_window(window_size, channel=3):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window
#计算SSIM
#直接使用SSIM的公式，但是在计算均值的时候，不是直接求像素平均值，而是采用归一化的高斯核来代替
#在计算方差和协方差时用到了公式Var（X）=E[X^2]-E[X]^2，cov(X,Y)=E[XY]-E[X]E[Y].
#上面求期望的操作采用高斯核卷积代替
def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
        self.channel = None
        self.window = None

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if (self.window is None) or (channel != self.channel) or (self.window.dtype != img1.dtype):
            self.window = create_window(self.window_size, channel=channel).to(img1.device).type(img1.dtype)
            self.channel = channel
        return ssim(img1, img2, window=self.window, window_size=self.window_size, size_average=self.size_average)

class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        # TODO: store window between calls if possible
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)

ssim_loss = msssim


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)



class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv = Sobelxy()

    def _rgb_to_y(self, x):
        # x: [B,3,H,W] -> Y: [B,1,H,W]，系数是常见的 ITU-R BT.601
        return (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])

    def forward(self, image_vis, image_ir, labels, generate_img, i):
        # 统一到单通道
        image_y = image_vis[:, :1, :, :]
        if image_ir.size(1) != 1:
            image_ir = image_ir[:, :1, :, :]
        if generate_img.size(1) == 3:
            generate_y = self._rgb_to_y(generate_img)
        else:
            generate_y = generate_img

        # （可选但推荐）保证值域一致，若你的图像已在 [0,1] 可略去
        image_y = torch.clamp(image_y, 0, 1)
        image_ir = torch.clamp(image_ir, 0, 1)
        generate_y = torch.clamp(generate_y, 0, 1)

        x_in_max = torch.max(image_y, image_ir)
        wb0 = 0.5
        wb1 = 0.5

        # 这里 msssim 支持 normalize=True，你的实现已兼容
        ssim_loss_temp1 = ssim_loss(generate_y, image_y, normalize=True)
        ssim_loss_temp2 = ssim_loss(generate_y, image_ir, normalize=True)
        ssim_value = wb0 * (1 - ssim_loss_temp1) + wb1 * (1 - ssim_loss_temp2)

        loss_in = F.mse_loss(x_in_max, generate_y)

        y_grad  = self.sobelconv(image_y)
        ir_grad = self.sobelconv(image_ir)
        gen_grad = self.sobelconv(generate_y)
        x_grad_joint = torch.max(y_grad, ir_grad)
        loss_grad = F.l1_loss(x_grad_joint, gen_grad)

        loss_total = (10 * ssim_value) + (10 * loss_in) + (1 * loss_grad)
        return loss_total, loss_in, ssim_value, loss_grad

#CT-MRI loss_in:10 loss_ssim:10,loss_grad:1

