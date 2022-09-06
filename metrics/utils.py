import torch
import numpy as np
from medpy.metric import hd, hd95, assd, ravd
from typing import Tuple

class Dice:

    def __init__(self):
        pass

    def __call__(self, output: torch.Tensor, target: torch.Tensor):
        output = output.flatten(start_dim=1)
        target = target.flatten(start_dim=1)
        inter = torch.sum(output * target, dim=1)
        # union = torch.sum(output * output, dim=1) + torch.sum(target * target, dim=1)
        union = torch.sum(torch.pow(output, 2), dim=1) + torch.sum(torch.pow(target, 2), dim=1)
        return ((2 * inter + 1)/ (union + 1)).mean(dim=0)


class DiceReverse:

    def __init__(self):
        pass

    def __call__(self, output: torch.Tensor, target: torch.Tensor):
        output = output.flatten(start_dim=1)
        target = target.flatten(start_dim=1)
        output = 1 - output
        target = 1 - target
        inter = torch.sum(output * target, dim=1)
        union = torch.sum(torch.pow(output, 2), dim=1) + torch.sum(torch.pow(target, 2), dim=1)
        return ((2 * inter + 1)/ (union + 1)).mean(dim=0)

class HD:

    def __init__(self) -> None:
        pass
    
    @torch.no_grad()
    def __call__(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        try:
            output = output.cpu().numpy()
            output = np.where(output > 0.5, 1, 0)
            target = target.cpu().numpy()
            return hd95(output, target)
        except:
            return torch.Tensor(0)

class Accuracy:

    def __init__(self) -> None:
        pass

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.clone()
        target = target.clone()

        pred = torch.where(pred > 0.5, 1, 0)
        return torch.sum(pred == target) / target.numel()

class PrecisionRecall:

    def __init__(self) -> None:
        pass

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pred = pred.clone()
        target = target.clone()

        pred = torch.where(pred > 0.5, 1, 0)
        inter = pred * target

        precision = torch.sum(inter) / torch.sum(pred)
        recall = torch.sum(inter) / torch.sum(target)
        return precision, recall

class AverageBoundaryDistance:
    """
    ABD
    """
    def __init__(self) -> None:
        pass

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor):
        pred = torch.where(pred > 0.5, 1, 0).squeeze().cpu().detach().numpy()
        target = target.squeeze().cpu().detach().numpy()
        return torch.mean(torch.as_tensor([assd(_pred, _target) for _pred, _target in zip(pred, target) if len(np.where(_pred > 0)[0]) > 0 and len(np.where(_target > 0)[0]) > 0]))
        # return assd(pred, target)

class RelativeAbsoluteVolumeDistance:
    """
    RAVD
    """
    def __init__(self) -> None:
        pass

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor):
        pred = pred.squeeze().cpu().detach().numpy()
        target = target.squeeze().cpu().detach().numpy()
        return torch.mean(torch.as_tensor([ravd(_pred, _target) for _pred, _target in zip(pred, target) if len(np.where(_pred > 0)[0]) > 0 and len(np.where(_target > 0)[0]) > 0]))
        # return ravd(pred, target)

