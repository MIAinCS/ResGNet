from typing import List, Optional
import torch
from torch import nn

SKIP_BIAS = True
SAMPLE_MOM = 1 # 还是1最好

class FirstBlock(nn.Module):

    def __init__(self, in_channel: int, out_channel: int, conv_block):
        super().__init__()
        assert in_channel == 1, in_channel

        self.conv1 = conv_block(in_channel, out_channel, inplace=True)

        self.conv2 = nn.Sequential(
            *[conv_block(out_channel, out_channel, inplace=True) for _ in range(1)]
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class DownBlock(nn.Module):

    def __init__(self, in_channel: int, out_channel: int, conv_block):
        super().__init__()

        self.down_sample = nn.Sequential(
            nn.AvgPool3d(kernel_size=2, stride=2),
            nn.InstanceNorm3d(in_channel, affine=True, momentum=SAMPLE_MOM),
            nn.ELU(inplace=True),
        )

        self.conv1 = conv_block(in_channel, out_channel, inplace=True)

        if in_channel != out_channel:
            self.skip = conv_block(in_channel, out_channel, inplace=True)
        else:
            self.skip = None

    def forward(self, x):
        x = self.down_sample(x)
        if self.skip:
            res = self.skip(x)
        else:
            res = x
        x = self.conv1(x)
        return torch.add(x, res)


class LastBlock(nn.Module):

    def __init__(self, in_channel: int, conv_block):
        super().__init__()

        self.conv1 = conv_block(in_channel, in_channel, inplace=True)
        self.act = nn.Sigmoid()

        self.last = nn.Sequential(
            nn.Conv3d(in_channel, 1, (1, 3, 3), 1, (0, 1, 1)),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.last(x)
        x = self.act(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, conv_block):
        super().__init__()

        self.conv1 = conv_block(in_channel, out_channel, inplace=True)

        if in_channel != out_channel:
            self.skip = conv_block(in_channel, out_channel, inplace=True)
        else:
            self.skip = None

    def forward(self, x):
        if self.skip:
            res = self.skip(x)
        else:
            res = x

        x = self.conv1(x)
        return torch.add(x, res)


class DownLayer(nn.Module):

    def __init__(self, num_pooling: int,  conv_block):
        super().__init__()
        self.num_pooling = num_pooling

        cur_channel = 1
        next_channel = 32
        self.expand = 4

        blocks = []
        blocks.append(FirstBlock(cur_channel, next_channel, conv_block))
        cur_channel, next_channel = next_channel, next_channel * self.expand

        for i in range(num_pooling):
            blocks.append(DownBlock(cur_channel, next_channel, conv_block))
            if i == num_pooling - 2:
                cur_channel, next_channel = next_channel, next_channel
            else:
                cur_channel, next_channel = next_channel, next_channel * self.expand

        self.out_channel = cur_channel
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        outputs = []
        for i in range(self.num_pooling+1):
            x = self.blocks[i](x)
            outputs.append(x)
        return outputs


class UpLayer(nn.Module):

    def __init__(self, num_pooling: int, in_channel: int, expand: int, conv_block):
        super().__init__()

        self.num_pooling = num_pooling

        cur_channel = in_channel
        next_channel = in_channel // expand

        blocks = []
        up_samples = []
        for i in range(num_pooling - 1):
            up_samples.append(nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.InstanceNorm3d(cur_channel, momentum=SAMPLE_MOM, affine=True),  
                nn.ELU(inplace=True),
            ))
            blocks.append(UpBlock(cur_channel * 2, next_channel, conv_block))
            cur_channel, next_channel = next_channel, next_channel // expand
        up_samples.append(nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.InstanceNorm3d(cur_channel, momentum=SAMPLE_MOM, affine=True),
            nn.ELU(inplace=True),
        ))
        blocks.append(LastBlock(cur_channel * 2, conv_block))
        self.blocks = nn.ModuleList(blocks)
        self.up_samples = nn.ModuleList(up_samples)

    def forward(self, inputs: List[torch.Tensor]):
        assert self.num_pooling > 0, self.num_pooling
        last: Optional[torch.Tensor] = None
        for i in range(self.num_pooling, -1, -1):
            if i == self.num_pooling:
                last = inputs[i]
            else:
                last = self.blocks[self.num_pooling - i -1](torch.cat([self.up_samples[self.num_pooling - 1 - i](last), inputs[i]], dim=1))
        return last


class VNet(nn.Module):

    def __init__(self, conv_block, num_pooling=4):
        super().__init__()

        self.num_pooling = num_pooling

        self.down_layer = DownLayer(self.num_pooling, conv_block)

        self.up_layer = UpLayer(
            self.num_pooling, self.down_layer.out_channel, self.down_layer.expand, conv_block)

    def forward(self, x):
        outputs = self.down_layer(x)
        return self.up_layer(outputs)
