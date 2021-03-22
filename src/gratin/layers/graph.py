from torch_geometric.data import Data
import torch
import numpy as np

from .data_tools import *

EMPTY_FIELD_VALUE = -999


class SimpleTrajData(Data):
    def __init__(self, raw_positions, graph_info={}, traj_info={}):

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
