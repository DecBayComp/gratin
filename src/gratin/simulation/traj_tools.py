import numpy as np
import sys
import os


def get_traj_scale(traj):
    diff = traj[1:] - traj[:-1]
    dr_2 = np.sum(diff ** 2, axis=1)
    scale = np.sqrt(np.mean(dr_2))
    if scale == 0:
        return 1.0
    else:
        return scale


def sample_sphere(dim):
    X = np.random.normal(0, 1, dim)
    X /= np.linalg.norm(X, ord=2)
    return X


def get_drift(traj, drift_amplitude):
    dim = traj.shape[1]
    drift_norm = drift_amplitude * get_traj_scale(traj)
    u = sample_sphere(dim)
    drift = drift_norm * u
    return drift, u


def add_drift(traj, drift):
    L, dim = traj.shape
    d_pos = np.repeat(np.reshape(drift, (1, -1)), L, axis=0)
    d_pos = d_pos * np.repeat(np.reshape(np.arange(L), (-1, 1)), dim, axis=1)
    return traj + d_pos


def add_drift_to_traj(traj, drift_amplitude=0.0):
    drift, direction = get_drift(traj, drift_amplitude)
    return add_drift(traj, drift), direction * drift_amplitude


def add_noise(traj, noise_factor):
    noise_amplitude = noise_factor * get_traj_scale(traj)
    delta = np.random.normal(0, noise_amplitude, traj.shape)
    return traj + delta


def traj_scale(positions, scale_name):
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


class HiddenPrints:
    """
    https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
