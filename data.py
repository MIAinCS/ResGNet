import os
from glob import glob
from typing import List, Tuple, Optional
from random import randint
import random
import numpy as np
import SimpleITK as sitk
import torch
import torchvision.transforms as T
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import Compose, TrivialAugmentWide

from augmentation.random_deform import produceRandomlyDeformedImage
from params import config

class MixupData(Dataset):
    def __init__(self, arr_files: List, seg_files: List, alpha = 0.4):
        super().__init__()
        assert seg_files is None or len(arr_files) == len(seg_files)

        self.beta = torch.distributions.Beta(alpha, alpha)

        self.arr_files = arr_files
        self.seg_files = seg_files

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        arr_file1 = self.arr_files[idx]
        seg_file1 = self.seg_files[idx]
        arr1 = sitk.GetArrayFromImage(sitk.ReadImage(arr_file1))
        seg1 = sitk.GetArrayFromImage(sitk.ReadImage(seg_file1))

        _idx = idx + randint(0, len(self.arr_files))
        arr_file2 = self.arr_files[_idx % len(self.arr_files)]
        seg_file2 = self.seg_files[_idx % len(self.seg_files)]
        arr2 = sitk.GetArrayFromImage(sitk.ReadImage(arr_file2))
        seg2 = sitk.GetArrayFromImage(sitk.ReadImage(seg_file2))

        arr1, seg1 = torch.as_tensor(arr1, dtype=torch.float32), torch.as_tensor(seg1, dtype=torch.float32)
        arr2, seg2 = torch.as_tensor(arr2, dtype=torch.float32), torch.as_tensor(seg2, dtype=torch.float32)

        lamb = self.beta.sample()
        arr = lamb * arr1 + (1. - lamb) * arr2
        seg = lamb * seg1 + (1. - lamb) * seg2
        arr = arr[None, :]
        seg = seg[None, :]
        return arr, seg
    
    def __len__(self):
        return len(self.arr_files)



class Data(Dataset):

    def __init__(self, arr_files: List, seg_files: List = None):
        super().__init__()
        assert seg_files is None or len(arr_files) == len(seg_files)

        self.arr_files = arr_files
        self.seg_files = seg_files


    def __getitem__(self, idx):
        arr_file = self.arr_files[idx]
        seg_file = self.seg_files[idx] if self.seg_files is not None else None
        arr = sitk.GetArrayFromImage(sitk.ReadImage(arr_file))
        # arr = (arr - arr.mean()) / arr.std() 
        if seg_file is not None:
            seg: np.ndarray = sitk.GetArrayFromImage(sitk.ReadImage(seg_file))
            # arr, seg = produceRandomlyDeformedImage(arr, seg)
            arr = arr[None, :]
            seg = seg[None, :]
            arr, seg = torch.as_tensor(arr, dtype=torch.float32), torch.as_tensor(seg, dtype=torch.float32)
            return arr, seg
        else:
            arr = torch.as_tensor(arr, dtype=torch.float32)
            arr = arr[None, :]
            return arr, os.path.basename(arr_file)

    def __len__(self):
        return len(self.arr_files)

class OneFoldDataModule(LightningDataModule):

    def __init__(self, train_data_root: str, test_root: str):
        super().__init__()

        train_files = glob(os.path.join(train_data_root, "*[0-9].mhd"))
        train_seg_files = [v.replace(".mhd", "_seg.mhd") for v in train_files]
        test_files = glob(os.path.join(test_root, "*.mhd"))

        train_data = Data(train_files, train_seg_files)
        self.test_data = Data(test_files)

        val_len = len(train_data) // 10
        train_len = len(train_data) - val_len
        self.train_data, self.val_data = random_split(
            train_data, 
            lengths=[train_len, val_len],
            generator=torch.Generator().manual_seed(66)
        )

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        print(f"Stage: {stage}")

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=1, shuffle=True, pin_memory=True, num_workers=4, prefetch_factor=2)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=1, shuffle=True, pin_memory=True, num_workers=4, prefetch_factor=2)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, prefetch_factor=2)

class DataModule(LightningDataModule):

    def __init__(self, data_root: str, fold_num: int, batch_size = 2, dim=3, isbi_root: str = None):
        super().__init__()

        self.batch_size = batch_size

        arr_files = glob(os.path.join(data_root, "*[0-9].mhd"))
        seg_files = [v.replace(".mhd", "_seg.mhd") for v in arr_files]
        assert len(arr_files) == len(seg_files)


        length = len(arr_files)
        test_len = length // config["fold"]
        train_len = length - test_len

        test_arrs = arr_files[(fold_num * test_len):((fold_num + 1) * test_len)]
        test_segs = seg_files[(fold_num * test_len):((fold_num + 1) * test_len)]

        train_arrs = list(filter(lambda v: v not in test_arrs, arr_files))
        train_segs = list(filter(lambda v: v not in test_segs, seg_files))

        if isbi_root:
            isbi_files = glob(os.path.join(isbi_root, "*[0-9].mhd"))
            isbi_seg_files = [v.replace(".mhd", "_seg.mhd") for v in isbi_files]
            train_arrs.extend(isbi_files)
            train_segs.extend(isbi_seg_files)
        
        if config["data"] == "mixup":
            self.train_data, self.test_data = MixupData(train_arrs, train_segs), Data(test_arrs, test_segs)
        else:
            self.train_data, self.test_data = Data(train_arrs, train_segs), Data(test_arrs, test_segs)


    def prepare_data(self):
        pass

    def setup(self, stage: str):
        print(f"stage: {stage}")

    def train_dataloader(self):
        train_data = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=4, prefetch_factor=4)
        return train_data

    def test_dataloader(self):
        test_data = DataLoader(self.test_data, batch_size=self.batch_size, num_workers=2)
        return test_data

    def val_dataloader(self):
        test_data = DataLoader(self.test_data, batch_size=self.batch_size, num_workers=2, shuffle=True)
        return test_data

class ISBIData(LightningDataModule):

    def __init__(self, isbi_root: str, category: str):
        super().__init__()

        data_root = os.path.join(isbi_root, category)
        arr_files = glob(os.path.join(data_root, "data", "*.mhd"))
        seg_files = [v.replace(os.path.join(data_root, "data"), os.path.join(data_root, "label")) for v in arr_files]

        data = Data(arr_files, seg_files)
        data_len = len(data)
        train_len = data_len // 10 * 8
        val_len = (data_len - train_len) // 2
        test_len = data_len - train_len - val_len
        self.train_data, self.val_data, self.test_data = random_split(data, [train_len, val_len, test_len])

    def prepare_data(self) -> None:
        return super().prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        print(f"Stage: {stage}")
        return super().setup(stage)

    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=True, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, num_workers=4, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, num_workers=4, pin_memory=True)