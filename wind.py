# env_effects.py
"""
Small utilities to configure wind and fluid properties in a MuJoCo simulation.

What this module provides
-------------------------
- `set_wind_constant(model, speed_mps, direction_deg=0.0, vertical_mps=0.0)`:
    Set a constant wind vector in world coordinates.
- `ensure_fluid_enabled(model, density=1.225, viscosity=1.8e-5)`:
    Ensure the environment (air) has positive density/viscosity so aerodynamic
    and wind-related forces are applied.

Conventions & Units
-------------------
- Speed and velocity components are in meters/second (m/s).
- `direction_deg` is the **horizontal** wind direction in degrees:
    0° → +X, 90° → +Y (right-handed world frame).
- `vertical_mps` sets the Z component (typically 0 for horizontal wind).
"""

import numpy as np
import mujoco


def set_wind_constant(model, speed_mps: float, direction_deg: float = 0.0, vertical_mps: float = 0.0):
    """
    Set a constant wind field (speed in m/s). By default this is a horizontal wind along +X.

    Parameters
    ----------
    model : mujoco.MjModel
        Target model whose `opt.wind` vector will be set.
    speed_mps : float
        Horizontal wind speed magnitude in m/s.
    direction_deg : float, default 0.0
        Horizontal wind direction in degrees (0 = +X, 90 = +Y).
    vertical_mps : float, default 0.0
        Vertical wind component in m/s (positive is +Z).

    """
    theta = np.deg2rad(direction_deg)
    wx = float(speed_mps) * np.cos(theta)
    wy = float(speed_mps) * np.sin(theta)
    wz = float(vertical_mps)
    model.opt.wind[:] = (wx, wy, wz)


def ensure_fluid_enabled(model, density: float = 1.225, viscosity: float = 1.8e-5):
    """
    Ensure positive environment properties so fluid and wind forces are applied.

    Parameters
    ----------
    model : mujoco.MjModel
        Target model whose `opt.density` and `opt.viscosity` may be updated.
    density : float, default 1.225
        Air density in kg/m^3 (≈ sea level at 15°C).
    viscosity : float, default 1.8e-5
        Dynamic viscosity of air in Pa·s.

    """
    if model.opt.density <= 0:
        model.opt.density = float(density)
    if model.opt.viscosity <= 0:
        model.opt.viscosity = float(viscosity)
