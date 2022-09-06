import torch
from torch.nn.modules import padding
from visdom import Visdom
from torch import nn
from .utils import BasicBlock2D, transpose_conv_block, BasicBlock, transpose_conv_block2D
from .resgnet import ResNextBlock
from params import config
import torch.nn.functional as F


class UNet(nn.Module):

    def __init__(self, in_channel = 1, visdom: Visdom = None, skip_connection = False, dim=3):
        super().__init__()

        self.dim = dim
        if dim == 3:
            conv = ResNextBlock
            transpose_conv = transpose_conv_block
            dropout = nn.Dropout3d
        elif dim == 2:
            conv = BasicBlock2D
            transpose_conv = transpose_conv_block2D
            dropout = nn.Dropout2d
        else:
            raise ValueError()

        self.idx = 0

        self.vis = visdom

        expand = 6

        self.kernel_size = 3
        self.padding = 1

        channel = 16
        self.encoder_0 = nn.Sequential(
            conv(in_channel, channel, kernel_size=self.kernel_size, stride=1, padding=self.padding),
            conv(channel, channel, kernel_size=self.kernel_size, stride=1, padding=self.padding),
            # test9
            # conv(channel, channel, kernel_size=self.kernel_size, stride=1, padding=self.padding),
            # test9
        )

        self.encoder_1 = nn.Sequential(
            conv(channel, channel, kernel_size=2, stride=2, padding=0),
            conv(channel, channel * expand, kernel_size=self.kernel_size, stride=1, padding=self.padding),
            # test9
            # conv(channel * expand, channel * expand, kernel_size=self.kernel_size, stride=1, padding=self.padding),
            # test9
        )
        channel *= expand

        self.encoder_2 = nn.Sequential(
            conv(channel, channel, kernel_size=2, stride=2, padding=0),
            conv(channel, channel * expand, kernel_size=self.kernel_size, stride=1, padding=self.padding),
            # test9
            # conv(channel * expand, channel * expand, kernel_size=self.kernel_size, stride=1, padding=self.padding),
            # conv(channel * expand, channel * expand, kernel_size=self.kernel_size, stride=1, padding=self.padding),
            # conv(channel * expand, channel * expand, kernel_size=self.kernel_size, stride=1, padding=self.padding),
            # test9
            nn.Dropout3d(config["dropout"])
        )
        channel *= expand

        self.encoder_3 = nn.Sequential(
            conv(channel, channel, kernel_size=2, stride=2, padding=0),
            # test9
            # conv(channel, channel, kernel_size=self.kernel_size, stride=1, padding=self.padding),
            # conv(channel, channel, kernel_size=self.kernel_size, stride=1, padding=self.padding),
            # conv(channel, channel, kernel_size=self.kernel_size, stride=1, padding=self.padding),
            # test9
            transpose_conv(channel, channel, kernel_size=2, stride=2, padding=0, output_padding=0),
            dropout(config["dropout"])
        )
        self.feature_ch = channel

        self.decoder_3 = nn.Sequential(
            conv(channel * 2, channel * 2, kernel_size=self.kernel_size, stride=1, padding=self.padding),
            conv(channel * 2, channel, kernel_size=self.kernel_size, stride=1, padding=self.padding),
            conv(channel, channel  // expand, kernel_size=self.kernel_size, stride=1, padding=self.padding),
            transpose_conv(channel // expand, channel // expand, kernel_size=2, stride=2, padding=0, output_padding=0),
            dropout(config["dropout"])
        )
        channel //= expand


        self.decoder_2 = nn.Sequential(
            conv(channel * 2, channel * 2, kernel_size=self.kernel_size, stride=1, padding=self.padding),
            conv(channel * 2, channel, kernel_size=self.kernel_size, stride=1, padding=self.padding),
            conv(channel, channel  // expand, kernel_size=self.kernel_size, stride=1, padding=self.padding),
            transpose_conv(channel // expand, channel // expand, kernel_size=2, stride=2, padding=0, output_padding=0),
            dropout(config["dropout"])
        )
        channel //= expand

        self.decoder_1 = nn.Sequential(
            conv(channel * 2, channel * 2, kernel_size=self.kernel_size, stride=1, padding=self.padding),
            conv(channel * 2, channel, kernel_size=self.kernel_size, stride=1, padding=self.padding),
            conv(channel, channel  // expand, kernel_size=self.kernel_size, stride=1, padding=self.padding),
            conv(channel // expand, channel // expand, kernel_size=self.kernel_size, stride=1, padding=self.padding),
        )
        channel //= expand

        if dim == 3:
            self.segment = nn.Sequential(
                nn.Conv3d(channel, 1, kernel_size=self.kernel_size, stride=1, padding=self.padding),
                # nn.Sigmoid()
                nn.Tanh()
            )
        else: # dim 2
            self.segment = nn.Sequential(
                nn.Conv2d(channel, 1, kernel_size=self.kernel_size, stride=1, padding=self.padding),
                # nn.Sigmoid()
                nn.Tanh()
            )


    def forward(self, x):
        out1 = self.encoder_0(x)
        out2 = self.encoder_1(out1)
        out3 = self.encoder_2(out2)
        _out = self.encoder_3(out3)

        out3 = self.decoder_3(torch.cat([_out, out3], dim=1))
        out2 = self.decoder_2(torch.cat([out3, out2], dim=1))
        out1 = self.decoder_1(torch.cat([out2, out1], dim=1))

        out = self.segment(out1)
        out = (out + 1) / 2

        # if not self.training:
        #     out = remove_all_but_the_largest_comonent(out)

        return out