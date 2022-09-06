from typing import List
from torch import nn
from torch import Tensor, einsum
import torch
from scipy.ndimage.morphology import distance_transform_edt as edt

class BoundaryLoss(nn.Module):
    """
    效果不是很好
    """
    def __init__(self):
        super().__init__()

    def __call__(self, output: Tensor, target: Tensor) -> Tensor:

        dist_maps = torch.from_numpy(edt(target.cpu().detach())).to(device=output.device)
        pc = output.type(torch.float32)
        dc = dist_maps.type(torch.float32)

        multipled = einsum("bcdwh,bcdwh->bcdwh", pc, dc)

        loss = multipled.mean()
        return loss