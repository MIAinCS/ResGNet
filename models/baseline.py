import torch
import torch.nn as nn

def block(in_channel, out_channel, kernel_size=(3,3,3), padding=(1,1,1)):
    return nn.Sequential(
                        nn.Conv3d(in_channel, out_channel, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                        nn.BatchNorm3d(out_channel),
                        nn.ReLU()
            )

class BaseResNext(nn.Module):
    
    def __init__(self, in_channel: int, out_channel: int, **kwargs):
        super().__init__()

        self.N_ch = 4

        self.skip = block(in_channel, out_channel)

        if in_channel > 4 and out_channel > 4:
            chs = []
            for i in range(self.N_ch):
                chs.append(
                    nn.Sequential(
                        block(in_channel, 4),
                        block(4, 4),
                        block(4, out_channel, kernel_size=(1, 1, 1), padding=(0, 0, 0))
                    )
                )
            self.chs = nn.ModuleList(chs)
        else:
            self.chs = block(in_channel, out_channel)


    def forward(self, input: torch.Tensor):
        skip = self.skip(input)
        if isinstance(self.chs, nn.ModuleList):
            out = None
            for ch in self.chs:
                if out is None:
                    out = ch(input)
                else:
                    out = out + ch(input)
        else:
            out = self.chs(input)
        out = out + skip
        return out


class InceptionV4(nn.Module):

    def __init__(self, in_channel: int, out_channel: int, **kwargs):
        super().__init__()

        self.N_ch = 4

        self.skip = block(in_channel, out_channel)

        if in_channel > 4 and out_channel > 4:
            chs = []
            for i in range(self.N_ch):
                chs.append(
                    nn.Sequential(
                        block(in_channel, 4, kernel_size=(1, 1, 1), padding=(0, 0, 0)),
                        block(4, 4),
                    )
                )
            self.chs = nn.ModuleList(chs)
            self.last = block(4 * self.N_ch, out_channel, kernel_size=(1, 1, 1), padding=(0, 0, 0))
        else:
            self.chs = block(in_channel, out_channel)

    def forward(self, input: torch.Tensor):
        skip = self.skip(input)
        if isinstance(self.chs, nn.ModuleList):
            out = torch.cat([self.chs[i](input) for i in range(self.N_ch)], dim=1)
            out = self.last(out)
        else:
            out = self.chs(input)

        out = out + skip
        return out