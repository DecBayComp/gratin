import pytorch_lightning as pl
from codecarbon import track_emissions
from gratin.models.network_lightning import MainNet
from gratin.data.data_classes import DataModule
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from gratin.training.callbacks import LatentSpaceSaver, Plotter


def setup(tasks, ds_params, net_params, dl_params, graph_info):
    pl.seed_everything(1234)
    dm = DataModule(ds_params=ds_params, dl_params=dl_params, graph_info=graph_info)
    dm.setup()
    model = MainNet(
        tasks=tasks, latent_dim=net_params["latent_dim"], n_c=net_params["n_c"], dm=dm
    )

    return model, dm


@track_emissions(project_name="unsupervised-rw")
def train_and_test(model, dm, logger):

    ES = EarlyStopping(
        monitor="train_loss",
        min_delta=0.001,
        patience=10,
        verbose=True,
        mode="min",
        strict=True,
    )
    LRM = LearningRateMonitor("epoch")
    tag = "unsupervised_v1"
    CKPT = ModelCheckpoint(
        dirpath="/gaia/models",
        filename=tag,
        monitor="train_loss",
        mode="min",
        verbose=True,
    )
    LSS = LatentSpaceSaver()

    trainer = pl.Trainer(
        auto_lr_find=False,
        gpus=1,
        auto_select_gpus=False,
        gradient_clip_val=1.0,
        log_gpu_memory="min_max",
        reload_dataloaders_every_epoch=True,
        callbacks=[ES, LRM, CKPT, LSS, Plotter()],
        accelerator="ddp",
        log_every_n_steps=150,
        flush_logs_every_n_steps=300,
        terminate_on_nan=True,
        track_grad_norm=2,
        weights_summary="full",
        profiler="simple",
        replace_sampler_ddp=True,
        logger=logger,
    )
    # trainer.tune(model=model, datamodule=dm)
    trainer.fit(model=model, datamodule=dm)
    trainer.test(model=model, datamodule=dm, verbose=True)


if __name__ == "__main__":
    ds_params = {
        "dim": 2,
        "RW_types": ["sBM", "CTRW", "fBM", "LW", "OU", "BM"],
        "N": int(1e5),
        "length_range": (10, 30),
        "noise_range": (0.2, 0.5),  # Issues when noise is too low
        "drift_range": (0.0, 0.5),
    }

    net_params = {"latent_dim": 32, "n_c": 32}

    dl_params = {"batch_size": 128, "num_workers": 10}

    graph_info = {
        "edges_per_point": 20,
        "clip_trajs": True,
        "scale_types": ["step_std", "step_mean", "pos_std"],
        "log_features": True,
        "edge_method": "geom_causal",
    }

    tasks = ["model", "drift_norm", "alpha"]

    model, dm = setup(tasks, ds_params, net_params, dl_params, graph_info)

    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir="/gaia/tb_logs",
        default_hp_metric=False,
    )
    train_and_test(model, dm, tb_logger)