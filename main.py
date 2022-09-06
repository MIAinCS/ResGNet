
from models.utils import BasicBlock2D
from visdom import Visdom
from datetime import datetime
import pandas as pd
from params import config
from models import UNet
from torch import nn
import torch
from net import Net
from data import DataModule, ISBIData
import yaml
from pytorch_lightning.utilities.seed import seed_everything, reset_seed
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
import os
from time import time
import warnings
import math
import pretty_errors

from models.vnet import VNet
from models.resgnet import ResGNet
from models.baseline import InceptionV4, BaseResNext
# warnings.filterwarnings("ignore")
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def weights_init(m):
    classname = m.__class__.__name__
    if classname.startswith('Conv'):
        if config["activation"] == "leaky_relu":
            nn.init.kaiming_normal_(m.weight.data, a=0.01, mode="fan_out", nonlinearity="leaky_relu")

        #     m.weight.data.normal_(std=1.0 / math.sqrt(m.weight.data.shape[1]))
        # nn.init.kaiming_normal_(m.weight.data, a=0.01 if config["activation"] == "leaky_relu" else 0, mode="fan_out", nonlinearity="relu")


def main(fold_num=1):
    exp = config["exp"]

    if not os.path.exists(f"output_{exp}"):
        os.mkdir(f"output_{exp}")
    dices = []
    fig_idx = 0
    for fold in range(fold_num):
        # 66 -> 88 -> 100(101, 104??) -> 
        seed_everything(100, workers=True)
        fold = fold if config["num_exp"] > 1 else 3
        print(f"[Fold Num]: {fold}")
        print(config)
        # 最差的是第3折, 观察ta可能更有效

        if config["isbi"] == 0:
            data_module = DataModule(
                data_root, 
                fold_num=fold, 
                batch_size=config["batch_size"] if config["dim"] != 2 else 1, dim=config["dim"],
            )
        else:
            data_module = ISBIData(
                isbi_root,
                category=config["category"]
            )

        if config["net"].lower() == "unet":
            model = UNet(dim=config["dim"])
        elif config["net"].lower() == "vnet":
            model = VNet(ResGNet)
            # model = VNet(BaseResNext)
            # model =VNet(InceptionV4)
        else:
            raise ValueError()

        try:
            if config["visdom"] == 1:
                visdom = Visdom(
                    port=8889, env=f"Type-{config['train']}-{fold}")
                if not visdom.check_connection(3):
                    visdom = None
            else:
                visdom = None
        except:
            visdom = None
        net = Net(model, visdom=visdom, fig_idx=fig_idx)
        # net.apply(weights_init)

        ckpt_model = ModelCheckpoint(
            dirpath=f"ckpt_{exp}", save_weights_only=True, filename=f"net-fold-{fold}")   # 保存最后一个epoch
        logger = CSVLogger("logs", f"{exp}")
        logger.log_graph(net)
        trainer = Trainer(
            gpus=1 if torch.cuda.device_count() > 0 else 0,
            max_epochs=config["epoch"],
            callbacks=[ckpt_model],
            benchmark=True,  # it's too slow to set as `False`
            logger=logger,
            deterministic=False,
            check_val_every_n_epoch=5,
            # precision=16,
            # amp_backend="apex",
            # amp_level="O2"
        )

        ckpt_path = f"ckpt_{exp}/net-fold-{fold}.ckpt"
        if config["train"] == 0:
            if os.path.exists(ckpt_path):
                net.load_state_dict(torch.load(ckpt_path)["state_dict"])
            trainer.test(net, datamodule=data_module)
        elif config["train"] == 1:
            trainer.fit(net, datamodule=data_module)
            trainer.test(net, datamodule=data_module, ckpt_path="best")
        else:
            if os.path.exists(ckpt_path):
                net.load_state_dict(torch.load(ckpt_path)["state_dict"])
            trainer.fit(net, datamodule=data_module)
            trainer.test(net, datamodule=data_module, ckpt_path="best")

        fig_idx = net.fig_idx

        dices.append([trainer.logged_metrics["dice"], trainer.logged_metrics["hd"],
                     trainer.logged_metrics["precision"], trainer.logged_metrics["recall"]])
        reset_seed()

    result = {}
    result.update(config)
    result.update({i: v for i, v in enumerate(dices)})
    dt = pd.DataFrame(columns=list(result.keys()))
    dt = dt.append(result, ignore_index=True)
    dt.to_csv(os.path.join(
        f"output_{exp}", f"{datetime.fromtimestamp(time()):%Y_%m_%d-%H_%M}.csv"), index=False)


if __name__ == "__main__":
    data_config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    data_root = data_config["processed"]["train"]
    isbi_root = data_config["processed"]["isbi"]

    main(config["num_exp"])
