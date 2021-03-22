import pytorch_lightning as pl
from pytorch_lightning.metrics.regression import MeanAbsoluteError as MAE
from pytorch_lightning.metrics import F1
from pytorch_lightning.metrics import ExplainedVariance as EV
import torch.nn as nn
import torch
from functools import partial
from .network_parts import MLP, AlphaPredictor, TrajsEncoder
from ..training.network_tools import L2_loss, Category_loss, is_concerned
from ..data.data_classes import DataModule
from torch.optim.lr_scheduler import ExponentialLR


class MainNet(pl.LightningModule):
    def __init__(
        self,
        tasks: list,
        dm: DataModule,
        n_c: int,  # number of convolutions
        latent_dim: int,
        n_final_convolutions: int = 1,
        params_scarcity: int = 0,
        gamma: float = 0.98,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.encoder = TrajsEncoder(n_c=n_c, latent_dim=latent_dim, dm=dm)

        self.save_hyperparameters("gamma")
        self.save_hyperparameters("lr")
        self.save_hyperparameters("tasks")
        self.save_hyperparameters("latent_dim")
        self.save_hyperparameters("n_c")
        self.save_hyperparameters(dm.ds_params)
        self.save_hyperparameters(dm.graph_info)

        outputs = self.get_output_modules(
            tasks, latent_dim, dm.ds_params["dim"], dm.ds_params["RW_types"]
        )

        self.out_networks = {}
        self.losses = {}
        self.targets = {}
        for out in outputs:
            net, target, loss = outputs[out]
            self.out_networks[out] = net
            self.targets[out] = target
            self.losses[out] = loss
        self.out_networks = nn.ModuleDict(self.out_networks)
        self.loss_scale = {}
        self.set_loss_scale()

        if "alpha" in tasks:
            self.MAE = MAE()
        if "model" in tasks:
            self.F1 = F1(len(dm.ds_params["RW_types"]))
        if "drift_norm" in tasks:
            self.EV = EV()

    def set_loss_scale(self):
        if "model" in self.losses:
            self.loss_scale["model"] = 1.0
        if "alpha" in self.losses:
            self.loss_scale["alpha"] = 1.5 / 12.0
        if "drift_norm" in self.losses:
            self.loss_scale["drift_norm"] = (
                self.hparams["drift_range"][1] - self.hparams["drift_range"][0]
            ) / 12.0

    def get_output_modules(self, tasks, latent_dim, dim, RW_types):
        outputs = {}
        if "alpha" in tasks:
            outputs["alpha"] = (
                AlphaPredictor(input_dim=latent_dim),
                partial(self.get_target, target="alpha"),
                L2_loss,
            )
        if "model" in tasks:
            outputs["model"] = (
                MLP([latent_dim, 2 * latent_dim, latent_dim, len(RW_types)]),
                partial(self.get_target, target="model"),
                Category_loss,
            )
        if "drift" in tasks:
            outputs["drift"] = (
                MLP([latent_dim, 2 * latent_dim, latent_dim, dim]),
                partial(self.get_target, target="drift"),
                L2_loss,
            )
        if "log_theta" in tasks:
            outputs["log_theta"] = (
                MLP([latent_dim, 2 * latent_dim, latent_dim, 1]),
                partial(self.get_target, target="log_theta"),
                L2_loss,
            )
        if "drift_norm" in tasks:
            outputs["drift_norm"] = (
                MLP([latent_dim, 2 * latent_dim, 1], last_activation=nn.ReLU()),
                partial(self.get_target, target="drift_norm"),
                L2_loss,
            )
        return outputs

    def get_target(self, data, target):
        if target == "alpha":
            return data.alpha
        elif target == "model":
            return data.model
        elif target == "drift":
            return data.drift
        elif target == "drift_norm":
            return data.drift_norm
        elif target == "log_theta":
            return data.log_theta
        else:
            raise NotImplementedError("Unknown target %s" % target)

    def forward(self, x):
        h = self.encoder(x)
        out = {}
        # targets = {}
        # losses = {}

        for net in self.out_networks:
            out[net] = self.out_networks[net](h)

        return out, h

    def training_step(self, batch, batch_idx):
        loss, out, targets = self.shared_step(batch, stage="train")
        return {"loss": loss, "preds": out, "targets": targets}

    def test_step(self, batch, batch_idx):
        loss, _, _ = self.shared_step(batch, stage="test")
        return loss

    def on_test_epoch_end(self):
        self.logger.log_hyperparams(self.hparams)

    def validation_step(self, batch, batch_idx):
        loss, out, targets = self.shared_step(batch, stage="val")
        return {"loss": loss, "preds": out, "targets": targets}

    def shared_step(self, batch, stage="train"):
        # Part of the computation that is common to train, val and test
        out, h = self(batch)
        targets = {}
        losses = {}
        weights = {}

        # Save h ?
        # Callback ?
        # Write to TB

        for net in out:

            targets[net] = self.targets[net](batch)
            w = is_concerned(targets[net])
            weights[net] = torch.mean(1.0 * w)
            # print(f"{net} : <w> = {torch.mean(1.*w)}")
            losses[net] = self.losses[net](out[net], targets[net], w)

            losses[net] /= self.loss_scale[net]
            self.log(
                "%s_%s_loss" % (net, stage),
                losses[net],
                on_step=False,
                on_epoch=True,
                logger=True,
            )

            if net == "alpha":
                self.log(
                    "%s_MAE" % stage,
                    self.MAE(out[net][w], targets[net][w]),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=stage == "val",
                    logger=True,
                )
            elif net == "model":
                self.log(
                    "%s_F1" % stage,
                    self.F1(out[net][w], targets[net][w].view(-1)),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=stage == "val",
                    logger=True,
                )
            elif net == "drift_norm":
                self.log(
                    "%s_EV" % stage,
                    self.EV(out[net][w], targets[net][w]),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=stage == "val",
                    logger=True,
                )

        out["latent"] = h
        # Pondération des loss en fonction du nombre de samples concernés
        loss = sum([losses[net] * weights[net] for net in losses])
        self.log(
            "%s_loss" % stage,
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss, out, targets

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), amsgrad=True, lr=self.hparams["lr"]
        )
        scheduler = ExponentialLR(optimizer, gamma=self.hparams["gamma"])
        return [optimizer], [scheduler]
