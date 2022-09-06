import SimpleITK as sitk
from metrics.clDice import soft_dice_cldice
from PIL import Image
import cv2
from datetime import datetime
import matplotlib.pyplot as plt
from pytorch_lightning import LightningModule
from visdom import Visdom
from models.unet import UNet
from metrics import (
    Dice, HD, DiceLoss, MCC_Loss, boundary_plot, PrecisionRecall, ActiveContourLoss, AreaLoss, HausdorffERLoss, HausdorffDTLoss, TopKLossTh, TverskyLoss, FocalTverskyLoss,
    AverageBoundaryDistance, RelativeAbsoluteVolumeDistance
)
from metrics.boundary_loss import BoundaryLoss
from metrics.loss import SoftDice
from torch import nn
from torch.nn.parameter import Parameter
from torch.optim import Adam, lr_scheduler, SGD, AdamW
from params import config
import torch
from typing import cast
import os
import numpy as np
from pytorch_lightning.utilities.metrics import metrics_to_scalars
import matplotlib as mpl
mpl.use("AGG")


class Net(LightningModule):

    def __init__(self, net, visdom: Visdom = None, save_fig=True, fig_idx=0, submit_mode=False):
        super().__init__()
        self.submit_mode = submit_mode

        if visdom is not None:
            self.vis = visdom
        else:
            self.vis = None

        # self.sigma1 = Parameter(torch.as_tensor(1., dtype=torch.float32))
        # self.sigma2 = Parameter(torch.as_tensor(1., dtype=torch.float32))

        self.net = net
        self.save_fig = save_fig
        # self.unet = UNet(visdom=self.vis)

        self.dice = Dice()
        self.hd = HD()
        self.precision_recall = PrecisionRecall()

        self.abd = AverageBoundaryDistance()
        self.ravd = RelativeAbsoluteVolumeDistance()

        # self.clDice = soft_dice_cldice()

        self.dice_loss = DiceLoss()
        # self.ce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor([2]))
        # self.ce_loss = nn.BCEWithLogitsLoss()
        # self.mcc_loss = MCC_Loss()
        # self.topK_loss = TopKLossTh(0.5)
        # self.boundary_loss = BoundaryLoss()
        # self.soft_dice_loss = SoftDice()
        # self.tversky_loss = TverskyLoss()
        self.focal_tversky_loss = FocalTverskyLoss()

        self.fig_idx = fig_idx

        self.lr = config["lr"]
        self.weight_decay = config["weight_decay"]
        self.warm_up = config["warm_up"]

        # self.automatic_optimization = False

        self.alpha = 1

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        if config["optim"] == "adam":
            optim1 = Adam(self.parameters(), lr=self.lr,
                          weight_decay=self.weight_decay)
            # optim1 = SGD(self.parameters(), lr=self.lr,
                        #  weight_decay=self.weight_decay, momentum=0.8)
        elif config["optim"] == "sgd":
            optim1 = Adam(self.parameters(), lr=self.lr,
                          weight_decay=self.weight_decay)
            optim2 = SGD(self.parameters(), lr=self.lr,
                         weight_decay=self.weight_decay, momentum=0.8)
        else:
            raise ValueError()
        # lr_sche = lr_scheduler.StepLR(optim, 100, gamma=0.9)
        # lr_sche1 = lr_scheduler.CosineAnnealingWarmRestarts(
        #     optim1, 4, 2, eta_min=1e-6)

        # 周期大了更稳定 
        lr_sche1 = lr_scheduler.CosineAnnealingLR(optim1, 7)

        return [{
                "optimizer": optim1,
                # "frequency": 50,
                "lr_scheduler": {
                    "scheduler": lr_sche1,
                    "monitor": "val_dice",
                    "interval": "epoch"
                }
                },
        ]

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        # warm up
        if self.trainer.global_step < self.warm_up:
            lr_scale = min(1.0, float(
                self.trainer.global_step + 1) / self.warm_up)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr

        optimizer.step(closure=optimizer_closure)

    def training_step(self, batch, batch_idx, optimizer_idx=-1):
        if not self.automatic_optimization:
            opt = self.optimizers()
            opt.zero_grad()
        arr, target = batch
        arr = cast(torch.Tensor, arr)
        out = self(arr)
        # if batch_idx % 4 == 0:

        dice = self.dice(out, target)
        precision, recall = self.precision_recall(out, target)

        cur_epoch = self.trainer.current_epoch

        # mcc_loss =  self.mcc_loss(out, target)
        dice_loss = self.dice_loss(out, target)
        # soft_dice_loss = self.soft_dice_loss(out, target)
        # vice_loss = self.boundary_loss(out, target)
        # area = self.area_loss(out, target)
        # hd_dt_loss = self.hd_dt(out, target)
        # vice_loss = self.topK_loss(out, target)
        # vice_loss = self.tversky_loss(out, target)
        vice_loss = self.focal_tversky_loss(out, target)
        # self.alpha = 1 - min(cur_epoch, 50) // 5 * 0.005
        self.alpha = 1 - cur_epoch // 5 * 0.01
        loss = dice_loss * self.alpha + vice_loss * (1 - self.alpha)

        if not self.automatic_optimization:
            self.manual_backward(loss)
            opt.step()

        self.show_out(out)
        self.show_out(arr, "grad")

        self.log_dict({
            "loss": loss,
            "lr": self.optimizers().state_dict()["param_groups"][0]['lr'],
            "dice": dice,
            "precision": precision,
            "recall": recall,
        }, prog_bar=True, on_step=True)

        return loss

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        if self.submit_mode:
            arr, mhd_name = batch
            out = self(arr)
            self.save_output(arr, None, out, mhd_name=mhd_name[0])
        else:
            arr, target = batch
            out = self(arr)

            if target is not None:
                dice = self.dice(out, target)
                hd = self.hd(out, target)
                precision, recall = self.precision_recall(out, target)
                abd = self.abd(out, target)
                ravd = self.ravd(out, target)

                start_edge = torch.where(target > 0)[2][0]
                end_edge = torch.where(target > 0)[2][-1]
                span = end_edge - start_edge + 1
                base_edge = span * 3 // 10
                mid_edge = span * 7 // 10
                base_idx = range(start_edge, start_edge + base_edge)
                mid_idx = range(start_edge + base_edge, start_edge + mid_edge)
                apex_idx = range(start_edge + mid_edge, end_edge)

                base_dice = self.dice(out[:, :, base_idx], target[:, :, base_idx])
                base_hd = self.hd(out[:, :, base_idx], target[:, :, base_idx])
                base_abd = self.abd(out[:, :, base_idx], target[:, :, base_idx])
                base_ravd = self.ravd(out[:, :, base_idx], target[:, :, base_idx])

                mid_dice = self.dice(out[:, :, mid_idx], target[:, :, mid_idx])
                mid_hd = self.hd(out[:, :, mid_idx], target[:, :, mid_idx])
                mid_abd = self.abd(out[:, :, mid_idx], target[:, :, mid_idx])
                mid_ravd = self.ravd(out[:, :, mid_idx], target[:, :, mid_idx])

                apex_dice = self.dice(out[:, :, apex_idx], target[:, :, apex_idx])
                apex_hd = self.hd(out[:, :, apex_idx], target[:, :, apex_idx])
                apex_abd = self.abd(out[:, :, apex_idx], target[:, :, apex_idx])
                apex_ravd = self.ravd(out[:, :, apex_idx], target[:, :, apex_idx])

                if hd is not None:
                    self.log_dict({
                        "dice": dice,
                        "hd": hd,
                        "precision": precision,
                        "recall": recall,
                        "abd": abd,
                        "ravd": ravd,
                    }, prog_bar=True)

                    self.log_dict({
                        "base_dice": base_dice,
                        "base_hd": base_hd,
                        "base_abd": base_abd,
                        "base_ravd": base_ravd,
                        "mid_dice": mid_dice,
                        "mid_hd": mid_hd,
                        "mid_abd": mid_abd,
                        "mid_ravd": mid_ravd,
                        "apex_dice": apex_dice,
                        "apex_hd": apex_hd,
                        "apex_abd": apex_abd,
                        "apex_ravd": apex_ravd
                    }, prog_bar=False)
                else:
                    self.log_dict({
                        "dice": dice,
                        "precision": precision,
                        "recall": recall,
                        "abd": abd,
                        "ravd": ravd,
                    }, prog_bar=True)

                    self.log_dict({
                        "base_dice": base_dice,
                        "base_hd": base_hd,
                        "base_abd": base_abd,
                        "base_ravd": base_ravd,
                        "mid_dice": mid_dice,
                        "mid_hd": mid_hd,
                        "mid_abd": mid_abd,
                        "mid_ravd": mid_ravd,
                        "apex_dice": apex_dice,
                        "apex_hd": apex_hd,
                        "apex_abd": apex_abd,
                        "apex_ravd": apex_ravd
                    }, prog_bar=False)
                if self.vis is not None:
                    with torch.no_grad():
                        global_step = self.trainer.global_step
                        self.vis.line([dice.cpu()], [global_step],
                                    win="test", name="dice", update="append")
                        self.vis.line([precision.cpu()], [global_step],
                                    win="test", name="precision", update="append")
                        self.vis.line([recall.cpu()], [global_step],
                                    win="test", name="recall", update="append")
            self.save_output(arr, target, out, dice.item())
        return batch_idx

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        arr, target = batch
        out = self(arr)
        dice = self.dice(out, target)
        hd = self.hd(out, target)

        if hd is not None:
            self.log_dict({
                "val_dice": dice,
                "val_hd": hd,
            }, prog_bar=True)
        else:
            self.log_dict({
                "val_dice": dice,
            }, prog_bar=True)
        return batch_idx

    @torch.no_grad()
    def show_out(self, out: torch.Tensor, name="out"):

        if name == "grad":
            if config["grad"] == 0:
                return
            origin = out.cpu().detach().numpy()[0, 0]
            grad = out.grad.cpu().detach().numpy()[0, 0]
            grad = (grad - grad.min()) / (grad.max() - grad.min()) * 255
            origin = (origin - origin.min()) / \
                (origin.max() - origin.min()) * 255
            grad = grad.astype(np.uint8)
            origin = origin.astype(np.uint8)
            for i, (arr, slice) in enumerate(zip(grad, origin)):
                arr = cast(torch.Tensor, arr)
                arr = np.asarray(Image.fromarray(arr).convert(
                    "RGB"), dtype=np.uint8)[:, :, ::-1]
                arr = cv2.applyColorMap(arr, cv2.COLORMAP_JET)
                slice = np.asarray(Image.fromarray(slice).convert(
                    "RGB"), dtype=np.uint8)[:, :, ::-1]
                plt.figure()
                plt.axis("off")
                result = slice * 0.5 + arr * 0.5
                result = (result - result.min()) / \
                    (result.max() - result.min())
                plt.imshow(result)
                plt.savefig(
                    f"grad_output/{datetime.now()}-{i}.png", bbox_inches="tight")
                plt.close()
            exit(0)
        else:
            if self.vis is None:
                return

            for i, arr in enumerate(out):
                arr = cast(torch.Tensor, arr)
                arr = arr.squeeze(dim=0).cpu().numpy()
                if self.net.dim == 3:
                    slice = arr[31]
                else:
                    slice = arr
                self.vis.heatmap(slice, win=f"{name}")
                break   # plot the first sample

    @torch.no_grad()
    def save_output(self, image: torch.Tensor, gt: torch.Tensor, pred: torch.Tensor, dice=None, mhd_name=None):
        exp = config["exp"]

        image = image.squeeze(dim=1).cpu().numpy()
        pred = pred.squeeze(dim=1).cpu().numpy()

        if self.submit_mode:
            assert mhd_name is not None
            for _pred in pred:
                if not os.path.exists("output_mhd"):
                    os.makedirs("output_mhd")      
                sitk.WriteImage(
                    sitk.GetImageFromArray(_pred), 
                    os.path.join("output_mhd", f"{mhd_name}"),
                )

            for _img, _pred in zip(image, pred):
                save_dir = os.path.join(f"output_{exp}", f"{self.fig_idx}")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                for i, (__img, __pred) in enumerate(zip(_img, _pred)):
                    boundary_plot(os.path.join(
                        save_dir, f"{i}"), __img, None, __pred, image_dice=None)
                self.fig_idx += 1
        else:
            if gt is not None:
                gt = gt.squeeze(dim=1).cpu().numpy()

                for _img, _gt, _pred in zip(image, gt, pred):

                    save_dir = os.path.join(f"output_{exp}", f"{self.fig_idx}")

                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir, exist_ok=True)
                    for i, (__img, __gt, __pred) in enumerate(zip(_img, _gt, _pred)):

                        boundary_plot(os.path.join(
                            save_dir, f"{i}"), __img, __gt, __pred, image_dice=dice)
                    

                    save_3d = os.path.join("output_3d", f"{self.fig_idx}")
                    if not os.path.exists(save_3d):
                        os.makedirs(save_3d, exist_ok=True)
                    binary_pred = _pred > 0.5
                    sitk.WriteImage(sitk.GetImageFromArray(_img * binary_pred), os.path.join(save_3d, "result.mhd"))
                    sitk.WriteImage(sitk.GetImageFromArray(_gt), os.path.join(save_3d, "gt.mhd"))

                    self.fig_idx += 1