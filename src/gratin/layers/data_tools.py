import numpy as np


def complete_graph(N):
    edge_start = np.array([i // N for i in range(N ** 2) if (i % N) != i // N])
    edge_end = np.array([i % N for i in range(N ** 2) if (i % N) != i // N])
    return edge_start, edge_end


def edges_uniform(N, D):
    delta = 1 + np.random.choice(N - 1, size=N * D, replace=True)
    edge_start = np.tile(np.arange(N), D)
    edge_end = (edge_start + delta) % N
    return edge_start, edge_end


def edges_geom_causal(N, D):
    # 0,0,0,...,N-1,N-1,N-1
    step_sizes = np.unique(np.floor(np.logspace(0, np.log10(N - 1), D)))
    # reduce mean degree to prevent doubled edges
    D = len(step_sizes)
    edge_start = np.repeat(np.arange(N), D)
    # A,B,C,...,A,B,C
    delta = -np.tile(step_sizes, N)
    edge_end = (edge_start + delta) % N

    return edge_start, edge_end


def traj_scale(positions: np.array, scale_name: str):
    dr = get_steps(positions)
    if scale_name == "step_std":
        scale = np.std(dr)
    elif scale_name == "step_mean":
        scale = np.mean(dr)
    elif scale_name == "step_sum":
        scale = np.sum(dr)
    elif scale_name == "pos_std":
        # mean of variances of position along each dimension
        scale = np.sqrt(np.mean(np.var(positions, axis=0)))
    else:
        raise NotImplementedError(f"Unknown scale {scale_name}")
    if scale == 0:
        scale = 1.0
    return scale


def get_steps(positions):
    dr = positions[1:] - positions[:-1]
    return np.sqrt(np.sum(dr ** 2, axis=1))
