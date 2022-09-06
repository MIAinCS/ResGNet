from torch import nn
from torch.nn.modules.activation import ELU
from torch.nn.modules.container import ModuleList
from params import config
from torch.nn import functional as F
import torch
import numpy as np
from scipy.ndimage import label
from .attention import CBAM

# norm = nn.InstanceNorm3d
norm = nn.GroupNorm
norm2d = nn.InstanceNorm2d


def get_groups(channel: int, out_channel: int = None):
    if out_channel is None:
        out_channel = channel
    for v in reversed([2 ** i for i in range(5)]):
        if channel % v == 0 and out_channel % v == 0:
            return v
    return channel

class FrozenBatchNorm3d(nn.Module):
    def __init__(self, n):
        super(FrozenBatchNorm3d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        # Cast all fixed parameters to half() if necessary
        if x.dtype == torch.float16:
            self.weight = self.weight.half()
            self.bias = self.bias.half()
            self.running_mean = self.running_mean.half()
            self.running_var = self.running_var.half()

        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1, 1)
        return x * scale + bias

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "{})".format(self.weight.shape[0])
        return s

class SEBlock(nn.Module):
    """
    像是内存不够了
    """
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        reduction = min(reduction, in_channels)
        self.linear1 = nn.Linear(in_channels, in_channels//reduction, bias=True)
        self.linear2 = nn.Linear(in_channels//reduction, in_channels)
        self.act = nn.ReLU(inplace=True)
        

    def forward(self, x):
        N, C, D, H, W = x.shape
        embedding = x.mean(dim=2).mean(dim=2).mean(dim=2)
        fc1 = self.act(self.linear1(embedding))
        fc2 = torch.sigmoid(self.linear2(fc1))
        return x * fc2.view(N, C, 1, 1, 1).contiguous()

def activation(*args, **kwargs):
    if config["activation"].lower() == "relu":
        return nn.ReLU(*args, **kwargs)
    elif config["activation"].lower() == "prelu":
        return nn.PReLU(*args, **kwargs)
    elif config["activation"].lower() == "leaky_relu":
        return nn.LeakyReLU(*args, **kwargs)
    elif config["activation"].lower() == "elu":
        return nn.ELU(*args, **kwargs)
    else:
        raise ValueError()

def conv_block(in_channel, channel, kernel_size=3, stride=1, padding=1, act=True, norm=True, inplace=False, groups=1):
    layers = [
        nn.Conv3d(in_channel, channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=True, groups=groups),
    ]
    if norm:
        layers.append(
            nn.InstanceNorm3d(channel, affine=True, momentum=0.4)
        )
    if act:
        layers.append(activation(inplace=inplace))
    return nn.Sequential(*layers)


def transpose_conv_block(in_channel, channel, kernel_size=3, stride=1, padding=1, output_padding=1):
    assert in_channel == channel
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="trilinear"),
        # nn.ConvTranspose3d(in_channel, channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, output_padding=output_padding),
        # norm(channel, affine=True),
        norm(8 if channel >= 8 else channel, channel, affine=True),
        activation()
    )

def transpose_conv_block2D(in_channel, channel, kernel_size=3, stride=1, padding=1, output_padding=1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channel, channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, output_padding=output_padding),
        norm2d(channel, affine=True),
        activation()
    )

class AtrousBlock(nn.Module):
    def __init__(self, in_channel: int, channel: int):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channel // 2, channel // 2, kernel_size=3, stride=1, dilation=8, padding="same")
        self.conv2 = nn.Conv3d(in_channel // 2, channel // 2, kernel_size=3, stride=1, dilation=1, padding="same")

        self.act1 = ELU(inplace=True)
        self.act2 = ELU(inplace=True)

        self.norm1 = norm(8 if channel // 2 >= 8 else channel // 2, channel // 2, affine=True)
        self.norm2 = norm(8 if channel // 2 >= 8 else channel // 2, channel // 2, affine=True)
    
    def forward(self, x: torch.Tensor):
        N, C, *_ = x.shape
        if C < 2 or C % 2 != 0:
            return x
        x1, x2 = torch.split(x, C // 2, dim=1)
        x1 = self.act1(self.norm1(self.conv1(x1)))
        x2 = self.act2(self.norm2(self.conv2(x2)))
        x = torch.cat([x1, x2], dim=1)
        return x


class BasicBlock(nn.Module):

    def __init__(self, in_channel, channel, kernel_size=3, stride=1, padding=1, mode="res2net"):
        """
        mode: ["resnet", "res2net"]
        """
        super().__init__()

        if stride == 1 and in_channel == channel:
            self.skip = None
        elif in_channel == channel:
            # self.skip = nn.Conv3d(in_channel, channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
            self.skip = nn.AvgPool3d(kernel_size=2, stride=2)
        else:
            self.skip = nn.Conv3d(in_channel, channel, kernel_size=1, stride=1, padding=0, groups=2 if in_channel % 2 == 0 and channel % 2 ==0 else 1, bias=False)
        
        self.conv1 = nn.Conv3d(in_channel, channel, kernel_size=kernel_size, stride=stride, padding=padding, groups=1, bias=False)
        # self.norm1  = norm(channel, affine=True)
        self.norm1 = norm(8 if channel >= 8 else channel, channel, affine=True)
        self.ac1 = activation()

        # self.conv2 = nn.Conv3d(channel, channel, kernel_size=3, stride=1, padding=1, groups=2 if in_channel % 2 == 0 and channel % 2 ==0 else 1, bias=False)
        self.conv2 = nn.Conv3d(channel, channel, kernel_size=3, stride=1, padding=1, groups=1, bias=False)
        # self.norm2 = norm(channel, affine=True)
        self.norm2 = norm(8 if channel >= 8 else channel, channel, affine=True)
        self.ac2 = activation()

        self.mode = mode
        if self.mode == "res2net" and channel >= 4:
            res2channel = channel // 4
            res2block = []
            for _ in range(3):
                res2block.append(nn.Sequential(
                    nn.Conv3d(res2channel, res2channel, 3, 1, 1, groups=res2channel),
                    FrozenBatchNorm3d(res2channel),
                    activation(),
                ))
            self.res2block = ModuleList(res2block)
        else:
            self.res2block = None
        
        # self.se_block = SEBlock(channel)
        # self.cbam = CBAM(channel, config["batch_size"])
        # self.ab = AtrousBlock(channel, channel)

    def forward(self, x):
        if self.skip is not None:
            skip = self.skip(x)
        else:
            skip = x

        x = self.conv1(x)
        # x = self.ab(x)
        x = self.norm1(x)
        x = self.ac1(x)

        if self.mode == "res2net" and self.res2block is not None:
            resx = torch.split(x, x.shape[1] // 4, dim=1)
            for i in range(3):
                if i == 0:
                    sp = resx[i]
                else:
                    sp = sp + resx[i]
                sp = self.res2block[i](sp)
                if i == 0:
                    out = sp
                else:
                    out = torch.cat([out, sp], dim=1)
            x = torch.cat([out, resx[3]], dim=1)
        
        x = self.conv2(x)
        # x = self.se_block(x)
        # x = self.cbam(x)
        x += skip
        x = self.norm2(x)
        x = self.ac2(x)

        return x


class BasicBlock2D(nn.Module):

    def __init__(self, in_channel, channel, kernel_size=3, stride=1, padding=1):
        super().__init__()

        if stride == 1 and in_channel == channel:
            self.skip = None
        else:
            self.skip = nn.Conv2d(in_channel, channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        
        self.conv1 = nn.Conv2d(in_channel, channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.inst_norm1  = norm2d(channel, affine=True)

        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.inst_norm2 = norm2d(channel, affine=True)

    def forward(self, x):
        if self.skip is not None:
            skip = self.skip(x)
        else:
            skip = x

        x = self.conv1(x)
        x = self.inst_norm1(x)
        x = F.leaky_relu(x)
        
        x = self.conv2(x)
        x += skip
        x = self.inst_norm2(x)
        x = F.leaky_relu(x)

        return x

def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]


@torch.no_grad()
def remove_all_but_the_largest_comonent(prob_map: torch.Tensor) -> torch.Tensor:
    """
    后处理，去除小的连通分量
    """
    for i in range(prob_map.shape[0]):
        mask = np.asarray((prob_map[i][0] > 0.5).cpu().detach(), dtype=np.int32)
        label_map, num_objects = label(mask)
        max_area = 0
        keep_id = -1
        for object_id in range(1, num_objects):
            object_map = label_map == object_id
            area = object_map.sum()
            if area > max_area:
                keep_id = object_id
                max_area = area
        keep_object = label_map == keep_id
        keep_object = torch.as_tensor(keep_object, dtype=prob_map.dtype, device=prob_map.device)
        prob_map[i][0] *= keep_object
    return prob_map


class Add(nn.Module):

    def __init__(self, channel):
        super().__init__()
        # self.norm = nn.InstanceNorm3d(channel, affine=True, momentum=0.4)
        self.act = nn.ELU(inplace=True)


    def forward(self, x1, x2):
        return self.act(torch.add(x1, x2))
        # return self.act(self.norm(torch.add(x1, x2)))
