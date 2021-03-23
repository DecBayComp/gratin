# Copyright 2019 by Gorka Munoz-Gil under the MIT license.
# This file is part of the Anomalous diffusion challenge (AnDi), and is
# released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included in the repository containing the file
# (github.com/gorkamunoz/ANDI)

""" 
This file contains a recopilatory of diffusion models used in ANDI. The class
is organized in three subclasses, each corresponding to a different dimensionality
of the output trajectory.

Currently the library containts the following models:
    Function   Dimensions    Description
    - bm          (1D)       Brownian motion
    - fbm      (1D/2D/3D)    Fractional browian motion, simulated by the fbm python library
    - ctrw     (1D/2D/3D)    Continuous time random walks
    - lw       (1D/2D/3D)    Levy walks
    - attm     (1D/2D/3D)    Annealed transit time
    - sbm      (1D/2D/3D)    Scaled brownian motion
        
Inputs of generator functions:
    - T (int): lenght of the trajectory. Gets transformed to int if input
                is float.
    - alpha (float): anomalous exponent
    
    Some generator functions also have optional inputs, see each function for
    details.
                            
Outputs:
    - numpy.array of lenght d.T, where d is the dimension
    
    Some generator functions have optional outputs, see each function for details
"""

import numpy as np
import fbm
from .utils_andi import regularize, bm1D, sample_sphere
from math import pi as pi
from scipy.special import erfcinv
from functools import partial

# from .traj_tools import HiddenPrints
import warnings
import jax.numpy as jnp
import jax


__all__ = ["diffusion_models"]


def sym_exp():
    """
    exponential on both sides
    """
    return np.random.exponential() * np.random.choice([-1.0, 1.0])


def triangle():
    return np.random.triangular(-1, 0, 1)


class diffusion_models(object):
    def __init__(self):
        """Constructor of the class"""

    class oneD:
        """Class cointaning one dimensional diffusion models"""

        def fbm(self, T, alpha):
            """ Creates a 1D fractional brownian motion trajectory"""
            H = alpha * 0.5
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return np.reshape(fbm.fbm(int(T - 1), H), (-1, 1))

        def ctrw(self, T, alpha, regular_time=True):
            """Creates a 1D continuous time tandom walk trajectory
            Optional parameters:
                :regular_time (bool):
                    - True if to transform the trajectory to regular time."""
            if alpha > 1:
                raise ValueError(
                    "Continuous random walks only allow for anomalous exponents <= 1."
                )
            # Generate the waiting times from power-law distribution
            times = np.cumsum((1 - np.random.rand(T)) ** (-1 / alpha))
            times = times[: np.argmax(times > T) + 1]
            # Generate the positions of the walk
            positions = np.cumsum(np.random.randn(len(times)))
            positions -= positions[0]
            # Output
            if regular_time:
                return np.reshape(regularize(positions, times, T), (-1, 1))
            else:
                return np.stack((times, positions))

        def lw(self, T, alpha):
            """ Creates a 1D Levy walk trajectory """
            if alpha < 1:
                raise ValueError("Levy walks only allow for anomalous exponents > 1.")
            # Define exponents for the distribution of flight times
            if alpha == 2:
                sigma = np.random.rand()
            else:
                sigma = 3 - alpha
            dt = (1 - np.random.rand(T)) ** (-1 / sigma)
            dt[dt > T] = T + 1
            # Define the velocity
            v = 10 * np.random.rand()
            # Generate the trajectory
            positions = np.empty(0)
            for t in dt:
                positions = np.append(
                    positions, v * np.ones(int(t)) * (2 * np.random.randint(0, 2) - 1)
                )
                if len(positions) > T:
                    break
            return np.reshape(np.cumsum(positions[: int(T)]) - positions[0], (-1, 1))

        def attm(self, T, alpha, regime=1):
            """Creates a 1D trajectory following the annealed transient time model
            Optional parameters:
                :regime (int):
                    - Defines the ATTM regime. Accepts three values: 0,1,2."""
            if regime not in [0, 1, 2]:
                raise ValueError("ATTM has only three regimes: 0, 1 or 2.")
            if alpha > 1:
                raise ValueError("ATTM only allows for anomalous exponents <= 1.")
            # Gamma and sigma selection
            if regime == 0:
                sigma = 3 * np.random.rand()
                gamma = np.random.uniform(low=-5, high=sigma)
                if alpha < 1:
                    raise ValueError(
                        "ATTM regime 0 only allows for anomalous exponents = 1."
                    )
            elif regime == 1:
                sigma = 3 * np.random.uniform(low=1e-2, high=1.1)
                gamma = sigma / alpha
                while sigma > gamma or gamma > sigma + 1:
                    sigma = 3 * np.random.uniform(low=1e-2, high=1.1)
                    gamma = sigma / alpha
            elif regime == 2:
                gamma = 1 / (1 - alpha)
                sigma = np.random.uniform(low=1e-2, high=gamma - 1)
            # Generate the trajectory
            positions = np.array([0])
            while len(positions) < T:
                Ds = (1 - np.random.uniform(low=0.1, high=0.99)) ** (1 / sigma)
                ts = Ds ** (-gamma)
                if ts > T:
                    ts = T
                positions = np.append(positions, positions[-1] + bm1D(ts, Ds))
            return np.reshape(positions[:T] - positions[0], (-1, 1))

        def sbm(self, T, alpha, sigma=1):
            """Creates a scaled brownian motion trajectory"""
            msd = (sigma ** 2) * np.arange(T + 1) ** alpha
            dx = np.sqrt(msd[1:] - msd[:-1])
            dx = np.sqrt(2) * dx * erfcinv(2 - 2 * np.random.rand(len(dx)))
            return np.reshape(np.cumsum(dx) - dx[0], (-1, 1))

        def ctrwExp(self, T, alpha, regular_time=True):
            """Creates a 1D continuous time tandom walk trajectory with exp step distribution
            Optional parameters:
                :regular_time (bool):
                    - True if to transform the trajectory to regular time."""
            if alpha > 1:
                raise ValueError(
                    "Continuous random walks only allow for anomalous exponents <= 1."
                )
            # Generate the waiting times from power-law distribution
            times = np.cumsum((1 - np.random.rand(T)) ** (-1 / alpha))
            times = times[: np.argmax(times > T) + 1]
            # Generate the positions of the walk
            shifts = [sym_exp() for _ in range(len(times))]
            positions = np.cumsum(shifts)
            positions -= positions[0]
            # Output
            if regular_time:
                return np.reshape(regularize(positions, times, T), (-1, 1))
            else:
                return np.stack((times, positions))

        def ctrwTri(self, T, alpha, regular_time=True):
            """Creates a 1D continuous time tandom walk trajectory with exp step distribution
            Optional parameters:
                :regular_time (bool):
                    - True if to transform the trajectory to regular time."""
            if alpha > 1:
                raise ValueError(
                    "Continuous random walks only allow for anomalous exponents <= 1."
                )
            # Generate the waiting times from power-law distribution
            times = np.cumsum((1 - np.random.rand(T)) ** (-1 / alpha))
            times = times[: np.argmax(times > T) + 1]
            # Generate the positions of the walk
            shifts = [triangle() for _ in range(len(times))]
            positions = np.cumsum(shifts)
            positions -= positions[0]
            # Output
            if regular_time:
                return np.reshape(regularize(positions, times, T), (-1, 1))
            else:
                return np.stack((times, positions))

        def BM(self, T, alpha, regular_time=True):
            """Creates a 1D Brownian Motion
            Optional parameters:
                :regular_time (bool):
                    - True if to transform the trajectory to regular time."""
            # Generate the waiting times from power-law distribution
            dt = 1.0
            dW = np.random.randn(T, 1) * np.sqrt(dt)
            X = np.cumsum(dW, axis=0)
            X -= X[0]
            return X

    class twoD(oneD):
        def ctrw(self, T, alpha, regular_time=True):
            """Creates a 2D continuous time tandom walk trajectory
            Optional parameters:
                :regular_time (bool):
                    - True if to transform the trajectory to regular time."""
            if alpha > 1:
                raise ValueError(
                    "Continuous random walks only allow for anomalous exponents <= 1."
                )
            # Generate the waiting times from power-law distribution
            times = np.cumsum((1 - np.random.rand(T)) ** (-1 / alpha))
            times = times[: np.argmax(times > T) + 1]
            # Generate the positions of the walk
            posX = np.cumsum(np.random.randn(len(times)))
            posY = np.cumsum(np.random.randn(len(times)))
            posX -= posX[0]
            posY -= posY[0]
            # Regularize and output
            if regular_time:
                regX = regularize(posX, times, T)
                regY = regularize(posY, times, T)
                # replace concat by stack
                return np.stack((regX, regY), axis=1)
            else:
                return np.stack((times, posX, posY))

        def fbm(self, T, alpha):
            """ Creates a 2D fractional brownian motion trajectory"""
            # Defin Hurst exponent
            H = alpha * 0.5
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return np.stack(
                    (fbm.fbm(int(T - 1), H), fbm.fbm(int(T - 1), H)), axis=1
                )

        def lw(self, T, alpha):
            """ Creates a 2D Levy walk trajectory """
            if alpha < 1:
                raise ValueError("Levy walks only allow for anomalous exponents > 1.")
            # Define exponents for the distribution of times
            if alpha == 2:
                sigma = np.random.rand()
            else:
                sigma = 3 - alpha
            dt = (1 - np.random.rand(T)) ** (-1 / sigma)
            dt[dt > T] = T + 1
            # Define the velocity
            v = 10 * np.random.rand()
            # Define the array where we save step length
            d = np.empty(0)
            # Define the array where we save the angle of the step
            angles = np.empty(0)
            # Generate trajectory
            for t in dt:
                d = np.append(
                    d, v * np.ones(int(t)) * (2 * np.random.randint(0, 2) - 1)
                )
                angles = np.append(
                    angles, np.random.uniform(low=0, high=2 * pi) * np.ones(int(t))
                )
                if len(d) > T:
                    break
            d = d[: int(T)]
            angles = angles[: int(T)]
            posX, posY = [d * np.cos(angles), d * np.sin(angles)]
            return np.stack(
                (np.cumsum(posX) - posX[0], np.cumsum(posY) - posY[0]), axis=1
            )

        def attm(self, T, alpha, regime=1):
            """Creates a 2D trajectory following the annealed transient time model
            Optional parameters:
                :regime (int):
                    - Defines the ATTM regime. Accepts three values: 0,1,2."""
            if regime not in [0, 1, 2]:
                raise ValueError("ATTM has only three regimes: 0, 1 or 2.")
            if alpha > 1:
                raise ValueError("ATTM only allows for anomalous exponents <= 1.")
            # Gamma and sigma selection
            if regime == 0:
                sigma = 3 * np.random.rand()
                gamma = np.random.uniform(low=-5, high=sigma)
                if alpha < 1:
                    raise ValueError(
                        "ATTM regime 0 only allows for anomalous exponents = 1."
                    )
            elif regime == 1:
                sigma = 3 * np.random.uniform(low=1e-2, high=1.1)
                gamma = sigma / alpha
                while sigma > gamma or gamma > sigma + 1:
                    sigma = 3 * np.random.uniform(low=1e-2, high=1.1)
                    gamma = sigma / alpha
            elif regime == 2:
                gamma = 1 / (1 - alpha)
                sigma = np.random.uniform(low=1e-2, high=gamma - 1)
            # Generate the trajectory
            posX = np.array([0])
            posY = np.array([0])
            while len(posX) < T:
                Ds = (1 - np.random.uniform(low=0.1, high=0.99)) ** (1 / sigma)
                ts = Ds ** (-gamma)
                if ts > T:
                    ts = T
                posX = np.append(posX, posX[-1] + bm1D(ts, Ds))
                posY = np.append(posY, posY[-1] + bm1D(ts, Ds))
            return np.stack((posX[:T] - posX[0], posY[:T] - posY[0]), axis=1)

        def sbm(self, T, alpha, sigma=1):
            """Creates a scaled brownian motion trajectory"""
            msd = (sigma ** 2) * np.arange(T + 1) ** alpha
            deltas = np.sqrt(msd[1:] - msd[:-1])
            dx = np.sqrt(2) * deltas * erfcinv(2 - 2 * np.random.rand(len(deltas)))
            dy = np.sqrt(2) * deltas * erfcinv(2 - 2 * np.random.rand(len(deltas)))
            return np.stack((np.cumsum(dx) - dx[0], np.cumsum(dy) - dy[0]), axis=1)

        def ctrwExp(self, T, alpha, regular_time=True):
            """Creates a 2D continuous time tandom walk trajectory
            Optional parameters:
                :regular_time (bool):
                    - True if to transform the trajectory to regular time."""
            if alpha > 1:
                raise ValueError(
                    "Continuous random walks only allow for anomalous exponents <= 1."
                )
            # Generate the waiting times from power-law distribution
            times = np.cumsum((1 - np.random.rand(T)) ** (-1 / alpha))
            times = times[: np.argmax(times > T) + 1]
            # Generate the positions of the walk
            shifts_x = [sym_exp() for _ in range(len(times))]
            shifts_y = [sym_exp() for _ in range(len(times))]
            posX = np.cumsum(shifts_x)
            posY = np.cumsum(shifts_y)
            posX -= posX[0]
            posY -= posY[0]
            # Regularize and output
            if regular_time:
                regX = regularize(posX, times, T)
                regY = regularize(posY, times, T)
                return np.stack((regX, regY), axis=1)
            else:
                return np.stack((times, posX, posY))

        def ctrwTri(self, T, alpha, regular_time=True):
            """Creates a 2D continuous time tandom walk trajectory
            Optional parameters:
                :regular_time (bool):
                    - True if to transform the trajectory to regular time."""
            if alpha > 1:
                raise ValueError(
                    "Continuous random walks only allow for anomalous exponents <= 1."
                )
            # Generate the waiting times from power-law distribution
            times = np.cumsum((1 - np.random.rand(T)) ** (-1 / alpha))
            times = times[: np.argmax(times > T) + 1]
            # Generate the positions of the walk
            shifts_x = [triangle() for _ in range(len(times))]
            shifts_y = [triangle() for _ in range(len(times))]
            posX = np.cumsum(shifts_x)
            posY = np.cumsum(shifts_y)
            posX -= posX[0]
            posY -= posY[0]
            # Regularize and output
            if regular_time:
                regX = regularize(posX, times, T)
                regY = regularize(posY, times, T)
                return np.stack((regX, regY), axis=1)
            else:
                return np.stack((times, posX, posY))

        def BM(self, T, alpha, regular_time=True):
            """Creates a 2D Brownian Motion
            Optional parameters:
                :regular_time (bool):
                    - True if to transform the trajectory to regular time."""
            # Generate the waiting times from power-law distribution
            assert alpha == 1
            dt = 1.0
            dW = np.random.randn(T, 2) * np.sqrt(dt)
            X = np.cumsum(dW, axis=0)
            X -= X[0]
            return X

    class threeD(oneD):
        def ctrw(self, T, alpha, regular_time=True):
            """Creates a 3D continuous time tandom walk trajectory
            Optional parameters:
                :regular_time (bool):
                    - True if to transform the trajectory to regular time."""
            if alpha > 1:
                raise ValueError(
                    "Continuous random walks only allow for anomalous exponents <= 1."
                )
            # Generate the waiting times from power-law distribution
            times = np.cumsum((1 - np.random.rand(T)) ** (-1 / alpha))
            times = np.append(0, times)
            times = times[: np.argmax(times > T) + 1]
            # Generate the positions of the walk
            lengths = np.random.randn(len(times))
            posX, posY, posZ = np.cumsum(sample_sphere(len(times), lengths), axis=1)
            posX = posX - posX[0]
            posY = posY - posY[0]
            posZ = posZ - posZ[0]
            # Regularize and output
            if regular_time:
                regX = regularize(posX, times, T)
                regY = regularize(posY, times, T)
                regZ = regularize(posZ, times, T)
                return np.stack((regX, regY, regZ), axis=1)
            else:
                return np.stack((times, posX, posY, posZ))

        def fbm(self, T, alpha):
            """ Creates a 3D fractional brownian motion trajectory"""
            # Define Hurst exponent
            H = alpha * 0.5
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return np.stack(
                    (
                        fbm.fbm(int(T - 1), H),
                        fbm.fbm(int(T - 1), H),
                        fbm.fbm(int(T - 1), H),
                    ),
                    axis=1,
                )

        def lw(self, T, alpha, regular_time=True):
            """ Creates a 3D Levy walk trajectory """
            if alpha < 1:
                raise ValueError("Levy walks only allow for anomalous exponents > 1.")
            # Define exponents for the distribution of times
            if alpha == 2:
                sigma = np.random.rand()
            else:
                sigma = 3 - alpha
            dt = (1 - np.random.rand(T)) ** (-1 / sigma)
            dt[dt > T] = T + 1
            # Define the velocity
            v = 10 * np.random.rand()
            # Create the trajectory
            posX = np.empty(0)
            posY = np.empty(0)
            posZ = np.empty(0)
            for t in dt:
                distX, distY, distZ = sample_sphere(1, v)
                posX = np.append(posX, distX * np.ones(int(t)))
                posY = np.append(posY, distY * np.ones(int(t)))
                posZ = np.append(posZ, distZ * np.ones(int(t)))
                if len(posX) > T:
                    break
            return np.stack(
                (
                    np.cumsum(posX[:T]) - posX[0],
                    np.cumsum(posY[:T]) - posY[0],
                    np.cumsum(posZ[:T]) - posZ[0],
                ),
                axis=1,
            )

        def attm(self, T, alpha, regime=1):
            """Creates a 3D trajectory following the annealed transient time model
            Optional parameters:
                :regime (int):
                    - Defines the ATTM regime. Accepts three values: 0,1,2."""
            if regime not in [0, 1, 2]:
                raise ValueError("ATTM has only three regimes: 0, 1 or 2.")
            if alpha > 1:
                raise ValueError("ATTM only allows for anomalous exponents <= 1.")
            # Parameter selection
            if regime == 0:
                sigma = 3 * np.random.rand()
                gamma = np.random.uniform(low=-5, high=sigma)
                if alpha < 1:
                    raise ValueError(
                        "ATTM Regime 0 can only produce trajectories with anomalous exponents = 1"
                    )
            elif regime == 1:
                sigma = 3 * np.random.uniform(low=1e-2, high=1.1)
                gamma = sigma / alpha
                while sigma > gamma or gamma > sigma + 1:
                    sigma = 3 * np.random.uniform(low=1e-2, high=1.1)
                    gamma = sigma / alpha
            elif regime == 2:
                gamma = 1 / (1 - alpha)
                sigma = np.random.uniform(low=1e-2, high=gamma - 1)
            # Create the trajectory
            posX = np.array([0])
            posY = np.array([0])
            posZ = np.array([0])
            while len(posX) < T:
                Ds = (1 - np.random.uniform(low=0.1, high=0.99)) ** (1 / sigma)
                ts = Ds ** (-gamma)
                if ts > T:
                    ts = T
                steps = np.sqrt(2 * Ds) * np.random.randn(int(ts))
                distX, distY, distZ = sample_sphere(len(steps), steps)
                posX = np.append(posX, posX[-1] + distX)
                posY = np.append(posY, posY[-1] + distY)
                posZ = np.append(posZ, posZ[-1] + distZ)
            return np.stack(
                (posX[:T] - posX[0], posY[:T] - posY[0], posZ[:T] - posZ[0]), axis=1
            )

        def sbm(self, T, alpha, sigma=1):
            """Creates a scaled brownian motion trajectory"""
            msd = (sigma ** 2) * np.arange(T + 1) ** alpha
            deltas = np.sqrt(msd[1:] - msd[:-1])
            dx = np.sqrt(2) * deltas * erfcinv(2 - 2 * np.random.rand(len(deltas)))
            dy = np.sqrt(2) * deltas * erfcinv(2 - 2 * np.random.rand(len(deltas)))
            dz = np.sqrt(2) * deltas * erfcinv(2 - 2 * np.random.rand(len(deltas)))
            return np.stack(
                (np.cumsum(dx) - dx[0], np.cumsum(dy) - dy[0], np.cumsum(dz) - dz[0]),
                axis=1,
            )

        def ctrwExp(self, T, alpha, regular_time=True):
            """Creates a 3D continuous time tandom walk trajectory
            Optional parameters:
                :regular_time (bool):
                    - True if to transform the trajectory to regular time."""
            # print("CTRW Exp")
            if alpha > 1:
                raise ValueError(
                    "Continuous random walks only allow for anomalous exponents <= 1."
                )
            # Generate the waiting times from power-law distribution
            times = np.cumsum((1 - np.random.rand(T)) ** (-1 / alpha))
            times = np.append(0, times)
            times = times[: np.argmax(times > T) + 1]
            # Generate the positions of the walk
            lengths = np.array([sym_exp() for _ in range(len(times))])
            posX, posY, posZ = np.cumsum(sample_sphere(len(times), lengths), axis=1)
            posX = posX - posX[0]
            posY = posY - posY[0]
            posZ = posZ - posZ[0]
            # Regularize and output
            if regular_time:
                regX = regularize(posX, times, T)
                regY = regularize(posY, times, T)
                regZ = regularize(posZ, times, T)
                return np.stack((regX, regY, regZ), axis=1)
            else:
                return np.stack((times, posX, posY, posZ))

        def ctrwTri(self, T, alpha, regular_time=True):
            """Creates a 3D continuous time tandom walk trajectory
            Optional parameters:
                :regular_time (bool):
                    - True if to transform the trajectory to regular time."""
            # print("CTRW Triangle")
            if alpha > 1:
                raise ValueError(
                    "Continuous random walks only allow for anomalous exponents <= 1."
                )
            # Generate the waiting times from power-law distribution
            times = np.cumsum((1 - np.random.rand(T)) ** (-1 / alpha))
            times = np.append(0, times)
            times = times[: np.argmax(times > T) + 1]
            # Generate the positions of the walk
            lengths = np.array([triangle() for _ in range(len(times))])
            posX, posY, posZ = np.cumsum(sample_sphere(len(times), lengths), axis=1)
            posX = posX - posX[0]
            posY = posY - posY[0]
            posZ = posZ - posZ[0]
            # Regularize and output
            if regular_time:
                regX = regularize(posX, times, T)
                regY = regularize(posY, times, T)
                regZ = regularize(posZ, times, T)
                return np.stack((regX, regY, regZ), axis=1)
            else:
                return np.stack((times, posX, posY, posZ))

        def BM(self, T, alpha, regular_time=True):
            # print("BM")
            """Creates a 3D Brownian Motion
            Optional parameters:
                :regular_time (bool):
                    - True if to transform the trajectory to regular time."""
            # Generate the waiting times from power-law distribution
            dt = 0.01
            dW = np.random.randn(T, 3) * np.sqrt(dt)
            X = np.cumsum(dW, axis=0)
            X -= X[0]
            return X


def generate_OU(D, T, log_theta, sigma):
    dt = 1e-4
    dW = np.random.randn(T - 1, D) * np.sqrt(dt)
    theta = np.power(10, log_theta)
    X = np.zeros((T, D))
    for i in range(1, T):
        X[i] = X[i - 1] + theta * (-X[i - 1]) * dt + sigma * dW[i - 1]
    return X


@jax.jit
def get_dx_cov(alpha, t):
    i = jnp.reshape(t, (-1, 1))
    j = i.T
    k = j - i
    C = 0.5 * (
        jnp.power(jnp.abs(k + 1), alpha)
        + jnp.power(jnp.abs(k - 1), alpha)
        - 2 * jnp.power(jnp.abs(k), alpha)
    )
    # C = jnp.where(k >= 100, 0., C)
    return C


@jax.jit
def get_fbm_from_cov(dx_cov, key):
    dx = jax.random.multivariate_normal(
        key=key, mean=jnp.zeros(dx_cov.shape[0]), cov=dx_cov
    )
    x = jnp.cumsum(dx, axis=0)
    return x


get_multidim_fbm = jax.vmap(get_fbm_from_cov, in_axes=(None, 0), out_axes=1)


@jax.jit
def log_likelihood(x, alpha):
    dx = jnp.diff(x, axis=0)
    C = get_dx_cov(alpha, jnp.arange(dx.shape[0]))
    mean = jnp.zeros(dx.shape[0])
    log_p = jax.scipy.stats.multivariate_normal.logpdf(dx, mean, C)
    return log_p


vLL = jax.jit(jax.vmap(jax.vmap(log_likelihood, in_axes=(1, None)), in_axes=(None, 0)))


@jax.jit
def ML_alpha(x, alpha_range=jnp.linspace(0.01, 1.99, 100)):
    LL = jnp.sum(vLL(x, alpha_range), axis=1)
    LL = jnp.where(jnp.isnan(LL), jnp.nanmin(LL), LL)
    i = jnp.argmax(LL)
    return alpha_range[i]


def get_fbm(D, alpha, T, key):
    dx_cov = get_dx_cov(alpha, jnp.arange(T))
    keys = jax.random.split(key, num=D)
    try:
        return get_multidim_fbm(dx_cov, keys)
    except Exception as e:
        print(e)
        raise


generators = {
    1: {
        "ATTM": diffusion_models().oneD().attm,
        "CTRW": diffusion_models().oneD().ctrw,
        "fBM": diffusion_models().oneD().fbm,
        "fBM_fullrange": diffusion_models().oneD().fbm,
        "fBM_jax": partial(get_fbm, D=1),
        "LW": diffusion_models().oneD().lw,
        "sBM": diffusion_models().oneD().sbm,
        "BM": diffusion_models().oneD().BM,
        "OU": partial(generate_OU, D=1),
    },
    2: {
        "ATTM": diffusion_models().twoD().attm,
        "CTRW": diffusion_models().twoD().ctrw,
        "fBM": diffusion_models().twoD().fbm,
        "LW": diffusion_models().twoD().lw,
        "sBM": diffusion_models().twoD().sbm,
        "BM": diffusion_models().twoD().BM,
        "OU": partial(generate_OU, D=2),
    },
    3: {
        "ATTM": diffusion_models().threeD().attm,
        "CTRW": diffusion_models().threeD().ctrw,
        "fBM": diffusion_models().threeD().fbm,
        "LW": diffusion_models().threeD().lw,
        "sBM": diffusion_models().threeD().sbm,
        "BM": diffusion_models().threeD().BM,
        "OU": partial(generate_OU, D=3),
    },
}


def params_sampler(model, seed=0):
    if model == "CTRW":
        return {"alpha": np.random.uniform(0.05, 1)}
    elif model == "LW":
        return {"alpha": np.random.uniform(1.0, 2.0)}
    elif model == "fBM":
        return {"alpha": np.random.uniform(0.05, 1.95)}
    elif model == "fBM_fullrange":
        return {"alpha": np.random.uniform(0.01, 1.99)}
    elif model == "sBM":
        return {"alpha": np.random.uniform(0.1, 2.0)}
    elif model == "OU":
        # log(Theta) = 2.5
        # La courbe du MSD se sépare de celle du BM à 10 pas
        return {"log_theta": np.random.uniform(2.6, 4), "sigma": 1.0}
    elif model == "ATTM":
        return {"alpha": np.random.uniform(0.05, 1.0)}
    elif model == "BM":
        return {"alpha": 1.0}
    else:
        raise NotImplementedError(f"Unknown model {model}")