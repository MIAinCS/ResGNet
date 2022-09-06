import torch
from torch import nn
from .utils import Dice, DiceReverse


class DiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.dice = Dice()

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        dice = self.dice(output, target)
        return 1 - dice


class SoftDice(nn.Module):

    def __init__(self):
        super().__init__()

        self.dice = Dice()
        self.dice_b = DiceReverse()

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        dice = self.dice(output, target)
        dice_b = self.dice_b(output, target)
        return 1 - (dice + dice_b) / 2
    

class MCC_Loss(nn.Module):
    """
    Calculates the proposed Matthews Correlation Coefficient-based loss.
    Args:
        inputs (torch.Tensor): 1-hot encoded predictions
        targets (torch.Tensor): 1-hot encoded ground truth
    """

    def __init__(self):
        super(MCC_Loss, self).__init__()

    def forward(self, inputs, targets):
        """
        MCC = (TP.TN - FP.FN) / sqrt((TP+FP) . (TP+FN) . (TN+FP) . (TN+FN))
        where TP, TN, FP, and FN are elements in the confusion matrix.
        """
        tp = torch.sum(torch.mul(inputs, targets))
        tn = torch.sum(torch.mul((1 - inputs), (1 - targets)))
        fp = torch.sum(torch.mul(inputs, (1 - targets)))
        fn = torch.sum(torch.mul((1 - inputs), targets))

        numerator = torch.mul(tp, tn) - torch.mul(fp, fn)
        denominator = torch.sqrt(
            torch.add(tp, 1, fp)
            * torch.add(tp, 1, fn)
            * torch.add(tn, 1, fp)
            * torch.add(tn, 1, fn)
        )

        # Adding 1 to the denominator to avoid divide-by-zero errors.
        mcc = torch.div(numerator.sum(), denominator.sum() + 1.0)
        return 1 - mcc


class AreaLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        resize_mat = torch.ones_like(target, device=target.device, dtype=target.dtype)
        xy_factor = torch.arange(0, target.shape[4])
        x_factor, y_factor = torch.meshgrid(xy_factor, xy_factor, indexing="xy")
        z_factor = torch.arange(0, target.shape[2])[:, None, None]

        X = resize_mat * x_factor.to(target.device)
        Y = resize_mat * y_factor.to(target.device)
        Z = resize_mat * z_factor.to(target.device)

        delta_X = torch.zeros_like(X, device=X.device)
        delta_Y = torch.zeros_like(Y, device=Y.device)
        delta_Z = torch.zeros_like(Z, device=Z.device)

        delta_X[:, :, :, :, 1:] = torch.pow(X[:, :, :, :, 1:] - X[:, :, :, :, :(target.shape[4]-1)], 2)
        delta_Y[:, :, :, 1:] = torch.pow(Y[:, :, :, 1:] - X[:, :, :, :(target.shape[3]-1)], 2)
        delta_Z[:, :, 1:] = torch.pow(Z[:, :, 1:] - Z[:, :, :(target.shape[2]-1)], 2)

        surface = (torch.sqrt(delta_X + delta_Y + delta_Z) * output).sum()

        return surface / target.numel()


class ActiveContourLoss(nn.Module):
    """
    https://github.com/lc82111/Active-Contour-Loss-pytorch/blob/master/Active-Contour-Loss.py 
    """

    def __init__(self):
        super().__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor, weight=10):
        """[summary]

        Args:
            output (torch.Tensor): (N, C, D, H, W)
            target (torch.Tensor): (N, C, D, H, W)
            weight (int, optional): [description]. Defaults to 10.

        Returns:
            [type]: [description]
        """

        # length term
        delta_d = output[:, :, 1:, :, :] - target[:, :, :-1, :, :]
        delta_h = output[:, :, :, 1:, :] - target[:, :, :, :-1, :]
        delta_w = output[:, :, :, :, 1:] - target[:, :, :, :, :-1] 

        delta_d = delta_d[:, :, 1:, :-2, :-2] ** 2
        delta_h = delta_h[:, :, :-2, 1:, :-2] ** 2
        delta_w = delta_w[:, :, :-2, :-2, 1:] ** 2
        delta_pred = torch.abs(delta_d + delta_h + delta_w)

        # where is a parameter to avoid square root is zero in practice.
        epsilon = 1e-8
        # eq.(11) in the paper, mean is used instead of sum.
        lenth = torch.mean(torch.sqrt(delta_pred + epsilon))

        # region term
        c_in = torch.ones_like(output)
        c_out = torch.zeros_like(output)

        # equ.(12) in the paper, mean is used instead of sum.
        region_in = torch.mean(output * (target - c_in)**2)
        region_out = torch.mean((1-output) * (target - c_out)**2)
        region = region_in + region_out

        loss = weight*lenth + region

        return loss


class BoundaryLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, gt: torch.Tensor):
        np_gt = gt.detach().cpu().numpy()
        from scipy.ndimage.morphology import distance_transform_edt as edt
        dt = edt(np_gt < 0.5)
        dt = torch.as_tensor(dt, dtype=torch.float32, device=gt.device)

        multi = torch.einsum("bcdhw,bcdhw->bcd", pred, dt)
        loss = multi.mean()
        return loss


class TverskyLoss(nn.Module):
    
    def __init__(self, alpha=0.7, beta=0.8):
        super().__init__()

        self.alpha = alpha
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        inter = torch.sum(pred * target)
        union = inter + self.alpha * torch.sum(pred * (1 - target)) + self.beta * torch.sum((1 - pred) * target)
        return 1 - inter / union

class FocalTverskyLoss(nn.Module):
    """
    good 3/7
    """
    def __init__(self, lambd=3/7):
        super().__init__()
        self.lambd = lambd
        self.tversky_loss = TverskyLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        return torch.pow(self.tversky_loss(pred, target), self.lambd)