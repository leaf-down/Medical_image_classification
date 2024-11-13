import torch
from torch.nn import nn


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batch_size, height, width, num_channels = x.size()
    channels_per_group = num_channels // groups
    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, height, width, groups, channels_per_group)
    x = torch.transpose(x, 3, 4).contiguous()
    # flatten
    x = x.view(batch_size, height, width, -1)
    return x


class eca_layer(nn.Module):
    def __init__(self, channel, k_size):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.k_size = k_size
        self.conv = nn.Conv1d(channel, channel, kernel_size=k_size, bias=False, groups=channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()  # b,c,h,w
        y = self.avg_pool(x)
        y = nn.functional.unfold(y.transpose(-1, -3), kernel_size=(1, self.k_size), padding=(0, (self.k_size - 1) // 2))
        y = self.conv(y.transpose(-1, -2)).unsqueeze(-1)
        y = self.sigmoid(y)
        x = x * y.expand_as(x)
        return x


class MIFA_block(nn.Module):
    def __init__(self, ch_1):
        super(MIFA_block, self).__init__()
        self.DWConv = nn.Conv2d(in_channels=ch_1, out_channels=ch_1, kernel_size=3, groups=ch_1, padding=1)
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.in_c = ch_1
        self.attention = eca_layer(channel=ch_1 * 2, k_size=1)

    def forward(self, l, g):
        Map_l = l
        Map_g = g
        l_map = self.DWConv(l)
        l_map = self.relu(l_map)
        l_map = self.sigmoid(l_map)

        g_map = self.DWConv(g)
        g_map = self.gelu(g_map)
        g_map = self.sigmoid(g_map)

        final_Map_g = g_map * Map_l
        final_Map_l = l_map * Map_g
        output = torch.cat((final_Map_l, final_Map_g), dim=1)  # B C H W
        output = output.permute(0, 2, 3, 1).contiguous()  # B H W C
        output = channel_shuffle(output, groups=self.in_c // 2)  # B H W C
        output = output.permute(0, 3, 1, 2).contiguous()  # B C H W
        output = self.attention(output)
        return output