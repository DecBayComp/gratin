import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from .callbacks import LatentSpaceSaver
from ..models.network_lightning import MainNet
from ..data.data_classes import DataModule


def setup_model_and_dm(tasks, ds_params, net_params, dl_params, graph_info):
    pl.seed_everything(1234)
    dm = DataModule(ds_params=ds_params, dl_params=dl_params, graph_info=graph_info)
    dm.setup()
    model = MainNet(
        tasks=tasks, latent_dim=net_params["latent_dim"], n_c=net_params["n_c"], dm=dm
    )

    return model, dm


def setup_trainer(logger, dirpath="/gaia/models", tag="default"):
    ES = EarlyStopping(
        monitor="val_loss",
        min_delta=0.001,
        patience=10,
        verbose=True,
        mode="min",
        strict=True,
    )
    LRM = LearningRateMonitor("epoch")
    CKPT = ModelCheckpoint(
        dirpath=dirpath, filename=tag, monitor="train_loss", verbose=True, mode="min"
    )
    LSS = LatentSpaceSaver()

    trainer = pl.Trainer(
        auto_lr_find=False,
        gpus=1 if torch.cuda.is_available() else 0,
        auto_select_gpus=False,
        gradient_clip_val=1.0,
        log_gpu_memory="min_max",
        reload_dataloaders_every_epoch=True,
        callbacks=[ES, LRM, CKPT, LSS],
        accelerator="ddp" if torch.cuda.is_available() else "ddp_cpu",
        log_every_n_steps=150,
        flush_logs_every_n_steps=300,
        terminate_on_nan=True,
        track_grad_norm=2,
        weights_summary="full",
        profiler="simple",
        logger=logger,
    )
    return trainer