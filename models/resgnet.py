import torch
from torch import nn
from .utils import conv_block
from params import config

def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class ResGNet(nn.Module):

    def __init__(self, in_channel, out_channel, **kwargs):
        super().__init__()

        self.in_channel = in_channel
        self.raw_in_channel = in_channel
        self.out_channel = out_channel
        self.raw_out_channel = out_channel
        self.kernel_style = [
            (1, 3, 3),
            (1, 1, 3),
            (1, 3, 1),
            (3, 3, 1),
        ]
        self.split_num = len(self.kernel_style)
        if self.in_channel % self.split_num == 0 and self.out_channel % self.split_num == 0:
            self.in_channel //= self.split_num
            self.out_channel //= self.split_num

        blocks = []
        residuals = []
        padding_style = [
            (k1 // 2, k2 // 2, k3 // 2) for k1, k2, k3 in self.kernel_style
        ]
        inplace = kwargs.get("inplace", True)

        for i in range(len(self.kernel_style)):
            cor = len(self.kernel_style) - i - 1
            blocks.append(nn.Sequential(
                conv_block(self.in_channel, self.in_channel, kernel_size=self.kernel_style[i], stride=1, padding=padding_style[i], inplace=inplace, norm=True, act=True),
                conv_block(self.in_channel, self.in_channel, kernel_size=self.kernel_style[cor], stride=1, padding=padding_style[cor], inplace=inplace, norm=True, act=True),
                conv_block(self.in_channel, self.out_channel, kernel_size=3, stride=1, padding=1, inplace=inplace, act=True, norm=True),
            ))
            if self.in_channel == self.out_channel:
                residuals.append(
                    Identity(),
                )
            else:
                residuals.append(
                    conv_block(self.in_channel, self.out_channel, kernel_size=1, stride=1, padding=0, act=True, norm=True)
                )

        self.scale = conv_block(self.out_channel * len(self.kernel_style), self.raw_out_channel, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), inplace=inplace, act=True, norm=True)
        self.blocks = nn.ModuleList(blocks)
        self.residuals = nn.ModuleList(residuals)
        
        if self.raw_in_channel != out_channel:
            self.skip = conv_block(self.raw_in_channel, self.raw_out_channel, kernel_size=1, stride=1, padding=0, act=True, norm=True)
        else:
            self.skip = None
    
    def forward(self, x):
        if self.skip:
            res = self.skip(x)
        else:
            res = x

        outputs = []
        if self.raw_in_channel % self.split_num != 0 or self.raw_out_channel % self.split_num != 0:
            for i in range(len(self.kernel_style)):
                if i % 2:
                    outputs.append(self.blocks[i](x) * self.residuals[i](x))
                else:
                    outputs.append(self.blocks[i](x) - self.residuals[i](x))
        else:
            splits = torch.split(x, self.raw_in_channel // self.split_num, dim=1)
            for i in range(len(self.kernel_style)):
                if i % 2:
                    outputs.append(self.blocks[i](splits[i]) * self.residuals[i](splits[i]))
                else:
                    outputs.append(self.blocks[i](splits[i]) + self.residuals[i](splits[i]))
        x = self.scale(torch.cat(outputs, dim=1))
        
        return torch.add(x, res)
        