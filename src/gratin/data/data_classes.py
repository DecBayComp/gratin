import torch_geometric.transforms as Transforms
from torch_geometric.data import Dataset, DataLoader
import numpy as np
from . import *
import pytorch_lightning as pl
from torch.utils.data import random_split
from torch.utils.data.distributed import DistributedSampler
from ..simulation.diffusion_models import generators, params_sampler
from ..simulation.traj_tools import *
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import torch
import numpy as np

from ..layers.data_tools import *

EMPTY_FIELD_VALUE = -999


class SimpleTrajData(Data):
    def __init__(
        self, raw_positions, graph_info={}, traj_info={}, original_positions=None
    ):
        if original_positions is None:
            original_positions = raw_positions
        dim = raw_positions.shape[1]
        default_traj_info = {
            "model": "unknown",
            "model_index": EMPTY_FIELD_VALUE,
            "drift_norm": EMPTY_FIELD_VALUE,
            "log_theta": EMPTY_FIELD_VALUE,
            "alpha": EMPTY_FIELD_VALUE,
            "noise": 0.0,
            "drift_vec": np.zeros(dim),
        }

        for key in default_traj_info:
            if default_traj_info[key] is None:
                continue
            if key not in traj_info:
                traj_info[key] = default_traj_info[key]

        raw_positions -= raw_positions[0]
        positions, clipped_steps = self.safe_positions(raw_positions, graph_info)
        X = self.get_node_features(positions, graph_info)
        edge_index = self.get_edges(X, positions, graph_info)
        E = self.get_edge_features(positions, edge_index, graph_info)

        reshape = lambda t: torch.reshape(torch.from_numpy(t), (1, -1))
        float_to_torch = lambda t: torch.Tensor([t]).view((1, 1))

        super(SimpleTrajData, self).__init__(
            pos=torch.from_numpy(positions).float(),
            original_pos=torch.from_numpy(original_positions).float(),
            clipped_steps=float_to_torch(clipped_steps),
            x=torch.from_numpy(X).float(),
            edge_index=edge_index,
            edge_attr=E,
            length=float_to_torch(positions.shape[0]),
            alpha=float_to_torch(traj_info["alpha"]),
            log_theta=float_to_torch(traj_info["log_theta"]),
            drift_norm=float_to_torch(traj_info["drift_norm"]),
            drift=reshape(traj_info["drift_vec"]),
            model=float_to_torch(traj_info["model_index"]).long(),
            noise=float_to_torch(traj_info["noise"]),
        )
        self.coalesce()

    def safe_positions(self, positions, graph_info):
        # clips too long jumps, returns the number of clipped steps
        if graph_info["clip_trajs"] == True:
            dr = get_steps(positions)
            M = np.median(dr) + 10 * np.std(dr)
            clipped_steps = np.sum(dr > M)
            dr_clipped = np.clip(dr, a_min=0.0, a_max=M)
            clipped_positions = np.zeros(positions.shape)
            deltas = positions[1:] - positions[:-1]
            deltas = deltas * np.reshape(dr_clipped / dr, (-1, 1))
            clipped_positions[1:] = np.cumsum(deltas, axis=0)
            return clipped_positions, clipped_steps
        else:
            return positions, 0

    def get_edges(self, X, positions, graph_info):

        D = graph_info["edges_per_point"]
        N = X.shape[0]
        if D >= N:
            edge_start, edge_end = complete_graph(N)
        if graph_info["edge_method"] == "uniform":
            edge_start, edge_end = edges_uniform(N, D)
        elif graph_info["edge_method"] == "geom_causal":
            edge_start, edge_end = edges_geom_causal(N, D)
        else:
            raise NotImplementedError(f"Method { graph_info['edge_method'] } not known")
        e = np.stack([edge_start, edge_end], axis=0)
        return torch.from_numpy(e).long()

    def get_node_features(self, positions, graph_info):
        features = []
        N = positions.shape[0]
        reshape = lambda a: np.reshape(a, (N - 1, -1))
        norm = lambda x: x
        if graph_info["log_features"]:
            norm = lambda x: np.log(1e-5 + x)
        # Time
        norm_time = np.arange(N - 1) / (N - 1)
        features.append(reshape(norm_time))
        # get displacements
        dr = get_steps(positions)
        dr_vec = positions[1:] - positions[:-1]

        MD = reshape(np.cumsum(dr) / (1.0 + np.arange(N - 1)))
        MSD = reshape(np.power(np.cumsum(dr ** 2), 1.0 / 2) / (1.0 + np.arange(N - 1)))
        MQD = reshape(np.power(np.cumsum(dr ** 4), 1.0 / 4) / (1.0 + np.arange(N - 1)))
        MaxD = reshape(np.maximum.accumulate(dr))

        for scale_name in graph_info["scale_types"]:
            scale = traj_scale(positions, scale_name)
            assert scale > 0
            if graph_info["position_features"]:
                # We can choose not to include position as features
                # This only makes sense if we use position differences in our convolutions
                features.append(np.clip(positions[:-1] / scale, -10, 10))
                features.append(np.clip(dr_vec / scale, -10, 10))
            features.append(norm(MD / scale))
            features.append(norm(MSD / scale))
            features.append(norm(MQD / scale))
            features.append(norm(MaxD / scale))
        features = np.concatenate(features, axis=1)
        assert np.sum(np.isnan(features)) == 0
        assert np.sum(np.isinf(features)) == 0
        return features

    def get_edge_features(self, positions, edge_index, graph_info):
        if graph_info["features_on_edges"] == False:
            return None
        else:
            raise NotImplementedError("Ne sait pas calculer les features sur les edges")
        return None


class TrajDataSet(Dataset):
    def __init__(
        self,
        N: int,
        dim: int,
        graph_info: dict,
        length_range: tuple,  # e.g = (5,50)
        noise_range: tuple,  # e.g. = (0.1,0.5),
        model_types: list,  # e.g. = ["fBM","CTRW"],
        drift_range: list,  # e.g. = (0.,0.3),
        seed_offset: int,  # e.g. = 0):
    ):
        self.N = N

        self.seed_offset = seed_offset
        self.graph_info = graph_info

        self.generators = generators[dim]
        self.model_types = model_types
        self.length_range = length_range
        self.drift_range = drift_range
        self.noise_range = noise_range
        self.dim = dim

        if self.graph_info["features_on_edges"] == False:
            transform = Transforms.ToSparseTensor()
        else:
            transform = None
        super(TrajDataSet, self).__init__(transform=transform)

    def len(self):
        return self.N

    def get_raw_traj(self, seed):
        np.random.seed(seed)

        model_index = np.random.choice(len(self.model_types))
        model = self.model_types[model_index]
        params = params_sampler(model, seed=seed)
        length = np.random.randint(low=self.length_range[0], high=self.length_range[1])
        raw_pos = self.generators[model](T=length, **params)
        return raw_pos, params, length, model, model_index

    def get_traj(self, seed):

        raw_pos, params, length, model, model_index = self.get_raw_traj(seed)

        if (self.drift_range[1] <= 0.0) or (
            model in ["OU", "BM"] and np.random.uniform() < 0.5
        ):
            # 50% du temps, on ne rajoute pas de drift aux OU et BM. Sinon, il associe le confinement au drift par exemple
            drifted_pos, drift_vec = raw_pos, np.zeros(raw_pos.shape[1])
            drift_norm = 0.0
        elif model in ["OU", "BM"]:
            drift_amplitude = np.random.uniform(
                self.drift_range[0], self.drift_range[1]
            )
            drifted_pos, drift_vec = add_drift_to_traj(
                raw_pos, drift_amplitude=drift_amplitude
            )
            drift_norm = np.linalg.norm(drift_vec)
        else:
            # Dans ce cas, ça ne fait pas sens d'ajouter du drift
            drifted_pos, drift_vec = (
                raw_pos,
                EMPTY_FIELD_VALUE * np.ones(raw_pos.shape[1]),
            )
            drift_norm = EMPTY_FIELD_VALUE

        if self.noise_range[1] > 0.0:
            noise_factor = np.random.uniform(self.noise_range[0], self.noise_range[1])
            noisy_pos = add_noise(drifted_pos, noise_factor=noise_factor)
        else:
            noisy_pos, noise_factor = drifted_pos, 0.0
        return (
            model,
            model_index,
            drift_vec,
            drift_norm,
            length,
            noise_factor,
            noisy_pos,
            raw_pos,
            params,
        )

    def make_plot(self):
        fig = plt.figure(figsize=(10, 10), dpi=150)

        for i in range(100):
            (
                model,
                model_index,
                drift_vec,
                drift_norm,
                length,
                noise_factor,
                noisy_pos,
                raw_pos,
                params,
            ) = self.get_traj(i)
            ax = fig.add_subplot(10, 10, i + 1)
            if self.dim == 1:
                ax.plot(noisy_pos[:, 0])
            else:
                ax.plot(noisy_pos[:, 0], noisy_pos[:, 1])
            if type(drift_norm) is not int:
                desc = f"{model} - D:{drift_norm:{1}.{1}} - N{noise_factor:{1}.{1}}"
            else:
                desc = f"{model} - A:{float(params['alpha']):{1}.{2}} - N{noise_factor:{1}.{1}}"
            ax.set_title(desc, fontsize=6)
            plt.axis("off")
        plt.tight_layout()
        return fig

    def get(self, idx):
        # pos = np.zeros((10,2))

        # if self.train:
        #    print("Train index %d" % idx)
        # else:
        #    print("Test index %d" % idx)
        seed = idx + self.seed_offset
        (
            model,
            model_index,
            drift_vec,
            drift_norm,
            length,
            noise_factor,
            noisy_pos,
            raw_pos,
            params,
        ) = self.get_traj(seed=seed)

        traj_info = params
        traj_info.update(
            {
                "model": model,
                "model_index": model_index,
                "drift_vec": drift_vec,
                "drift_norm": drift_norm,
                "length": length,
                "noise": noise_factor,
                "seed": seed,
            }
        )
        return SimpleTrajData(
            noisy_pos,
            graph_info=self.graph_info,
            traj_info=traj_info,
            original_positions=raw_pos,
        )


class ExpTrajDataSet(Dataset):
    def __init__(
        self,
        dim: int,
        graph_info: dict,
        trajs: list,
    ):  # e.g. = 0):
        self.N = len(trajs)
        self.trajs = trajs
        self.graph_info = graph_info
        self.dim = dim

        if self.graph_info["features_on_edges"] == False:
            transform = Transforms.ToSparseTensor()
        else:
            transform = None
        super(ExpTrajDataSet, self).__init__(transform=transform)

    def len(self):
        return self.N

    def make_plot(self):
        fig = plt.figure(figsize=(10, 10), dpi=150)

        for i in range(min(100, len(self))):
            noisy_pos = self.trajs[i]
            ax = fig.add_subplot(10, 10, i + 1)
            if self.dim == 1:
                ax.plot(noisy_pos[:, 0])
            else:
                ax.plot(noisy_pos[:, 0], noisy_pos[:, 1])
            plt.axis("off")
        plt.tight_layout()
        return fig

    def get(self, idx):

        noisy_pos = self.trajs[idx]
        traj_info = {"index": idx}

        return SimpleTrajData(
            noisy_pos, graph_info=self.graph_info, traj_info=traj_info
        )


class DataModule(pl.LightningDataModule):
    def __init__(self, ds_params, dl_params, graph_info):
        super().__init__()
        self.ds_params = ds_params
        self.dl_params = dl_params
        self.batch_size = dl_params["batch_size"]
        self.epoch_count = 0

        self.graph_info = graph_info
        for key in default_graph_info:
            if key not in self.graph_info:
                self.graph_info[key] = default_graph_info[key]
        self.round = 0

    def setup(self, stage=None):
        if stage is None:
            print("stage is None, strange...")
        N = self.ds_params["N"]
        dim = self.ds_params["dim"]
        noise_range = self.ds_params["noise_range"]
        model_types = self.ds_params["RW_types"]
        drift_range = self.ds_params["drift_range"]
        length_range = self.ds_params["length_range"]

        if stage == "fit" or stage is None:
            ds = TrajDataSet(
                N=N,
                dim=dim,
                noise_range=noise_range,
                model_types=model_types,
                length_range=length_range,
                drift_range=drift_range,
                graph_info=self.graph_info,
                seed_offset=N * self.epoch_count,
            )
            self.ds_train, self.ds_val = random_split(ds, [9 * (N // 10), N // 10])
        if stage == "test" or stage is None:
            ds = TrajDataSet(
                N=N,
                dim=dim,
                noise_range=noise_range,
                model_types=model_types,
                length_range=length_range,
                drift_range=drift_range,
                graph_info=self.graph_info,
                seed_offset=N * (self.epoch_count + 1),
            )
            self.ds_test = ds

        data = ds[0]
        self.x_dim = data.x.shape[1]
        try:
            self.e_dim = data.edge_attr.shape[1]
        except:
            self.e_dim = 0
        if self.round == 0:
            self.traj_examples = ds.make_plot()
        self.round += 1

    def train_dataloader(self):
        # print(f"Call train_dataloader for the {self.epoch_count}th time")
        self.setup(stage="fit")
        self.epoch_count += 1
        return DataLoader(
            self.ds_train,
            num_workers=self.dl_params["num_workers"],
            batch_size=self.batch_size,
            sampler=DistributedSampler(self.ds_train),
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            num_workers=self.dl_params["num_workers"],
            batch_size=self.batch_size,
            sampler=DistributedSampler(self.ds_val),
            pin_memory=True,
        )

    def test_dataloader(self, no_parallel=False):
        self.setup(stage="test")
        if no_parallel:
            # Otherwise, error Default process group is not initialized
            return DataLoader(
                self.ds_test,
                num_workers=self.dl_params["num_workers"],
                batch_size=self.batch_size,
                pin_memory=True,
            )
        else:
            return DataLoader(
                self.ds_test,
                num_workers=self.dl_params["num_workers"],
                batch_size=self.batch_size,
                sampler=DistributedSampler(self.ds_test),
                pin_memory=True,
            )
