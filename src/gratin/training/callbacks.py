from pytorch_lightning.callbacks import Callback
from pytorch_lightning.metrics.functional import confusion_matrix
from .network_tools import is_concerned
from collections import defaultdict
from tqdm import tqdm
from ..data.data_classes import EMPTY_FIELD_VALUE
import os, sys
import torch
import numpy as np
import matplotlib.pyplot as plt


class Plotter(Callback):
    def on_init_end(self, trainer):
        self.max_capacity = int(1e4)
        self.tb = trainer.logger.experiment
        self.round = 0

    def on_validation_epoch_start(self, trainer, pl_module):
        self.preds = defaultdict(lambda: [])
        self.info = defaultdict(lambda: [])
        self.n_items = 0

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if self.n_items > self.max_capacity:
            return
        targets = outputs["targets"]
        preds = outputs["preds"]

        for i, param in enumerate(targets):
            self.n_items += targets[param].shape[0]
            self.info[param].append(targets[param])
            self.preds[param].append(preds[param])

    def on_validation_epoch_end(self, trainer, pl_module):

        for param in tqdm(self.info, leave=False, colour="blue"):
            if param not in ["alpha", "drift_norm", "model"]:
                continue
            info = torch.cat(self.info[param], dim=0)
            pred = torch.cat(self.preds[param], dim=0)

            if param == "model":
                info = info[:, 0]
                pred = torch.argmax(pred, dim=1)
                n_models = len(pl_module.hparams["RW_types"])
                CM = (
                    confusion_matrix(pred, info, n_models, normalize="true")
                    .detach()
                    .cpu()
                )

                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.imshow(CM, cmap="Blues", vmin=0.0, vmax=1.0)
                ax.set_xticks(np.arange(n_models))
                ax.set_xticklabels(pl_module.hparams["RW_types"])
                ax.set_yticks(np.arange(n_models))
                ax.set_yticklabels(pl_module.hparams["RW_types"])

                self.tb.add_figure("model_val", fig, global_step=self.round, close=True)

            elif param == "drift_norm":
                cond = is_concerned(info).detach().cpu().numpy()
                info = info[:, 0].detach().cpu().numpy()
                pred = pred[:, 0].detach().cpu().numpy()
                info[~cond] = 0.0

                plt.figure()
                plt.hist(pred[~cond])

                fig = plt.figure()
                ax = fig.add_subplot(211)
                ax.scatter(info[cond], pred[cond], label="BM & OU with drift", s=3)
                ax.plot([0.0, 0.5], [0.0, 0.5], ls=":", c="red")
                ax.set_xlim((-0.05, 0.55))
                ax.set_ylim((-0.05, 0.55))
                ax.set_xlabel("True drift")
                ax.set_ylabel("Inferred drift")
                ax.legend()

                ax = fig.add_subplot(212)
                ax.hist(
                    pred[~cond],
                    bins=30,
                    density=True,
                    label="Anomalous diffusion",
                    range=(-0.05, 1.0),
                    alpha=0.5,
                )
                ax.hist(
                    pred[cond & (info <= 0.25)],
                    bins=30,
                    density=True,
                    label="BM & OU w/ low drift",
                    range=(-0.05, 1.0),
                    alpha=0.5,
                )
                ax.hist(
                    pred[cond & (info > 0.25)],
                    bins=30,
                    density=True,
                    label="BM & OU w/ high drift",
                    range=(-0.05, 1.0),
                    alpha=0.5,
                )
                ax.set_xlabel("Inferred drift")
                ax.legend()

                plt.tight_layout()

                self.tb.add_figure("DNorm_val", fig, global_step=self.round, close=True)

        self.round += 1


class LatentSpaceSaver(Callback):
    def on_init_end(self, trainer):
        self.max_capacity = int(5e3)  # TB samples down to 5e3 points anyway
        self.tb = trainer.logger.experiment
        self.round = 0

    def on_validation_epoch_start(self, trainer, pl_module):
        self.latent_vectors = []
        self.info = defaultdict(lambda: [])
        self.n_items = 0

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if self.n_items > self.max_capacity:
            return
        h = outputs["preds"]["latent"]
        targets = outputs["targets"]

        self.n_items += h.shape[0]
        for param in targets:
            self.info[param].append(targets[param])
        self.latent_vectors.append(h)

    def on_validation_epoch_end(self, trainer, pl_module):
        self.round += 1

        if not self.round % 10 == 0:
            return

        H = torch.cat(self.latent_vectors, dim=0).detach().cpu().numpy()
        for param in tqdm(self.info, leave=False, colour="blue"):
            if param not in ["alpha", "drift_norm", "model"]:
                continue
            info = torch.cat(self.info[param], dim=0)
            if param == "model":
                info = [pl_module.hparams["RW_types"][i] for i in info[:, 0]]
            else:
                info = [
                    "%.2f" % i if i != EMPTY_FIELD_VALUE else 0.0 for i in info[:, 0]
                ]
            try:
                self.tb.add_embedding(
                    H, metadata=info, tag=param, global_step=self.round
                )
            except AttributeError as e:
                print(e)
                print(
                    "Conflict between tensorboard and tensorflow (installed with umap-learn...)"
                )
                break

    def on_test_end(self, trainer, pl_module):
        metric_dict = dict()
        if "model" in pl_module.losses:
            metric_dict["model_F1"] = trainer.logged_metrics["test_F1"].item()
        if "alpha" in pl_module.losses:
            metric_dict["alpha_MAE"] = trainer.logged_metrics["test_MAE"].item()
        if "drift_norm" in pl_module.losses:
            metric_dict["drift_norm_EV"] = trainer.logged_metrics["test_EV"].item()
        hparams = dict()
        for p in pl_module.hparams:
            value = pl_module.hparams[p]
            if type(value) is tuple:
                if len(value) == 2:
                    hparams["%s_min" % p] = value[0]
                    hparams["%s_max" % p] = value[1]
            elif type(value) is list:
                for list_item in value:
                    hparams["has_%s" % list_item] = True
            else:
                hparams[p] = value
        self.tb.add_hparams(hparams, metric_dict, run_name="final")
        self.tb.flush()