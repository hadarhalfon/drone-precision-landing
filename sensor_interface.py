# sensor_interface.py
"""
sensor_interface.py
===================
Thin sensor layer that reads "true" states from MuJoCo via `mujoco_data`
and (optionally) adds zero-mean Gaussian noise for robustness testing.

Design
------
- A single, module-wide NumPy Generator is used for reproducible noise.
- Each `read_*` function calls the corresponding accessor in `mujoco_data`
  and then passes the result through `_add_noise(raw, sensor_noise)`.

Noise model
-----------
- Additive Gaussian:  N(0, sigma^2) applied element-wise.
- Set `sensor_noise=0.0` (default) to return noise-free readings.
"""
from __future__ import annotations
import numpy as np
import mujoco_data as md

# Single RNG for the entire module (reproducible by default)
_RNG = np.random.default_rng(42)

def _add_noise(arr, sigma: float):
    """
        Add element-wise Gaussian noise with std `sigma` to `arr`.

        Parameters
        ----------
        arr : array-like
            Input vector/array (e.g., position [x,y,z] or 6D velocity [wx,wy,wz,vx,vy,vz]).
        sigma : float
            Standard deviation of the additive noise. If <= 0, returns `arr` unchanged.

        Returns
        -------
        numpy.ndarray
            Noisy array with the same shape as the input.
        """
    a = np.asarray(arr, dtype=float)
    if float(sigma) <= 0.0:
        return a
    return a + _RNG.normal(0.0, float(sigma), size=a.shape)

# ===== Read APIs with optional noise =====
def read_drone_position(model, data, sensor_noise: float = 0.0):
    """
    Read the drone world position [x, y, z] and optionally add Gaussian noise.

    Parameters
    ----------
    model, data : MuJoCo handles
    sensor_noise : float, default 0.0
        Stddev of additive noise (meters).

    Returns
    -------
    numpy.ndarray shape (3,)
    """
    raw = md.get_drone_position(model, data)
    return _add_noise(raw, sensor_noise)

def read_platform_position(model, data, sensor_noise: float = 0.0):
    """
    Read the platform world position [x, y, z] and optionally add Gaussian noise.

    Parameters
    ----------
    model, data : MuJoCo handles
    sensor_noise : float, default 0.0
        Stddev of additive noise (meters).

    Returns
    -------
    numpy.ndarray shape (3,)
    """
    raw = md.get_platform_position(model, data)
    return _add_noise(raw, sensor_noise)

def read_drone_velocity(model, data, sensor_noise: float = 0.0):
    """
    Read the drone 6D velocity and optionally add Gaussian noise.

    Format: [wx, wy, wz, vx, vy, vz] where angular is first (rad/s),
    linear is last (m/s), expressed in the world frame (as provided by `mujoco_data`).

    Parameters
    ----------
    model, data : MuJoCo handles
    sensor_noise : float, default 0.0
        Stddev of additive noise (rad/s for angular entries, m/s for linear entries).

    Returns
    -------
    numpy.ndarray shape (6,)
    """
    raw = md.get_drone_velocity(model, data)
    return _add_noise(raw, sensor_noise)

def read_platform_velocity(model, data, sensor_noise: float = 0.0):
    """
    Read the platform 6D velocity and optionally add Gaussian noise.

    Format: [wx, wy, wz, vx, vy, vz] (world frame), as provided by `mujoco_data`.

    Parameters
    ----------
    model, data : MuJoCo handles
    sensor_noise : float, default 0.0
        Stddev of additive noise (rad/s for angular entries, m/s for linear entries).

    Returns
    -------
    numpy.ndarray shape (6,)
    """
    raw = md.get_platform_velocity(model, data)
    return _add_noise(raw, sensor_noise)
