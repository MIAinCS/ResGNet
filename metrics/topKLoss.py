from torch import nn
import torch.functional as F
import torch

class TopKLossTh(nn.Module):

    def __init__(self, th: float = 0.2):
        super().__init__()

        self.th = th
        # self.ce = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([0.5]))
        self.ce = nn.BCELoss()

    def forward(self, out: torch.Tensor, target: torch.Tensor):
        _target = target[((out > self.th)&(target == 0)) | ((out < self.th)&(target == 1))]
        _out = out[((out > self.th)&(target == 0)) | ((out < self.th)&(target == 1))]
        return self.ce(_out, _target)
