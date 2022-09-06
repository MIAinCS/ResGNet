import torch
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
matplotlib.use("AGG")

def boundary_plot(save_file: str, image: np.ndarray, gt: np.ndarray, pred: np.ndarray = None, image_dice = None):
    if isinstance(image, torch.Tensor):
        image = np.asarray(image)
    if gt is not None and isinstance(gt, torch.Tensor):
        gt = np.asarray(gt) 
    if pred is not None and isinstance(pred, torch.Tensor):
        pred = np.asarray(pred)
        pred = pred > 0.5
    image = (image - image.min()) / (image.max() - image.min()) * 255
    image = np.asarray(Image.fromarray(image).convert("RGB"))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # ground truth
    # image[..., 1] = np.where(gt == 1, 255, image[..., 1])

    # predication
    if pred is not None:
        pred = np.where(pred > 0.5, 1, 0)
        pred = pred.astype(np.uint8)
        if gt is not None:
            gt = gt.astype(np.uint8)
            contours, _ = cv2.findContours(gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, (0, 255, 0), 1, lineType=cv2.LINE_8)

        contours, _ = cv2.findContours(pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0, 0, 255), 1, lineType=cv2.LINE_8)


    image = image[:, :, ::-1]  # RGB

    plt.figure(figsize=(10, 10))
    if image_dice:
        plt.title(f"{image_dice}")
    plt.axis("off")
    plt.imshow(image)
    plt.savefig(f"{save_file}.png", bbox_inches="tight")
    plt.close()