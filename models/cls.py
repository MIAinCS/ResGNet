import torch 
from torch import nn
from utils import BasicBlock


class CliceCls(nn.Module):

    def __init__(self):
        super().__init__()

        module = []
        channel = 1
        next_channel = 16
        for i in range(5):
            module.append(nn.Sequential(
                BasicBlock(channel, next_channel, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
                BasicBlock(next_channel, next_channel, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            ))
            channel = next_channel
            next_channel = 2 * next_channel
        self.module = nn.ModuleList(module)

        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, arr: torch.Tensor, target: torch.Tensor):
        target = target.max(dim=3)[0].max(dim=3)[0]

        out: torch.Tensor = self.module(arr)
        out = out.mean(dim=3).mean(dim=3)

        return out, target


