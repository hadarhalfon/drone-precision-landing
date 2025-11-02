"""
calculator.py
=============
Lightweight math helpers used across the project.

Conventions & Units
-------------------
- Positions are world-frame [x, y, z] in meters (m).
- Velocities are world-frame [vx, vy, vz] in meters/second (m/s).
- Many helpers accept any iterable convertible to a 1D NumPy array.

Notes about Z offsets
---------------------
`DRONE_BOTTOM_FROM_ORIGIN_Z` and `PLATFORM_TOP_FROM_ORIGIN_Z` must match your
MuJoCo XML model. They describe the vertical offsets from each body's origin to:
- the **lowest** point of the drone (bottom of legs/props), and
- the **top** surface of the platform (landing plane),
respectively. If you change meshes/origins in the XML, update these values here.

"""
from __future__ import annotations
import numpy as np
import math

DRONE_BOTTOM_FROM_ORIGIN_Z =0.013
PLATFORM_TOP_FROM_ORIGIN_Z = 0.0507

def as_xyz(vec):
    """
        Return a 3-vector [x, y, z] from a vector-like input.

        If `vec` has 6 or more elements (e.g., [x,y,z,vx,vy,vz]), this returns the
        **last** three elements; otherwise it returns the first three.

        Parameters
        ----------
        vec : array-like
            Iterable convertible to a 1D float array.

        Returns
        -------
        numpy.ndarray
            Shape (3,) float array [x, y, z].

        Raises
        ------
        ValueError
            If the input has fewer than 3 elements.
        """
    a = np.asarray(vec,dtype = float).reshape(-1)
    if a.size >=6:
        return a[-3:].copy()
    if a.size >=3:
        return a[:3].copy()
    raise ValueError("Vector must have at least 3 elements for [x,y,z].")

def calculate_distance(d_pos,p_pos):
    """
        Horizontal (XY-plane) Euclidean distance between drone and platform centers.

        Parameters
        ----------
        d_pos, p_pos : array-like
            Position vectors with at least 3 elements each.

        Returns
        -------
        float
            sqrt((dx)^2 + (dy)^2) in meters.
        """
    d = as_xyz(d_pos)
    p = as_xyz(p_pos)
    dx = d[0] - p[0]
    dy = d[1] - p[1]
    return math.hypot(dx,dy)

def calculate_distance_x(d_pos,p_pos):
    """
        Signed horizontal distance along X: (drone_x - platform_x).

        Parameters
        ----------
        d_pos, p_pos : array-like
            Position vectors with at least 3 elements each.

        Returns
        -------
        float
            Delta in meters along +X.
        """
    d = as_xyz(d_pos)
    p = as_xyz(p_pos)
    return float(d[0] - p[0])

def calculate_distance_y(d_pos,p_pos):
    """
       Signed horizontal distance along Y: (drone_y - platform_y).

       Parameters
       ----------
       d_pos, p_pos : array-like
           Position vectors with at least 3 elements each.

       Returns
       -------
       float
           Delta in meters along +Y.
       """
    d = as_xyz(d_pos)
    p = as_xyz(p_pos)
    return float(d[1] - p[1])

def calculate_altitude(d_pos,p_pos):
    """
        Altitude-to-land: vertical clearance from the drone's **lowest** point
        to the platform **top** surface. Positive means the drone is above the top.

        Formula
        -------
        drone_bottom_z = d_pos[2] - DRONE_BOTTOM_FROM_ORIGIN_Z
        platform_top_z = p_pos[2] + PLATFORM_TOP_FROM_ORIGIN_Z
        altitude_to_land = drone_bottom_z - platform_top_z

        Parameters
        ----------
        d_pos, p_pos : sequence
            Position-like sequences where index 2 is Z (meters).

        Returns
        -------
        float
            Altitude-to-land in meters (positive above, zero at contact, negative below).
        """

    drone_bottom_z = d_pos[2] - DRONE_BOTTOM_FROM_ORIGIN_Z
    platform_top_z = p_pos[2] + PLATFORM_TOP_FROM_ORIGIN_Z
    altitude_to_land = drone_bottom_z - platform_top_z

    return float(altitude_to_land)

def calculate_relative_velocity_x(d_vel,p_vel):
    """
        Relative velocity along X: v_rel_x = v_drone_x - v_platform_x.

        Parameters
        ----------
        d_vel, p_vel : array-like
            Velocity vectors with at least 3 elements each (m/s).

        Returns
        -------
        float
            Relative x-velocity in m/s.
        """
    dv = as_xyz(d_vel)
    pv = as_xyz(p_vel)
    return float(dv[0] - pv[0])

def calculate_relative_velocity_y(d_vel,p_vel):
    """
        Relative velocity along Y: v_rel_y = v_drone_y - v_platform_y.

        Parameters
        ----------
        d_vel, p_vel : array-like
            Velocity vectors with at least 3 elements each (m/s).

        Returns
        -------
        float
            Relative y-velocity in m/s.
        """
    dv = as_xyz(d_vel)
    pv = as_xyz(p_vel)
    return float(dv[1] - pv[1])

