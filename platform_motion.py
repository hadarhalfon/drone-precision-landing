# platform_motion.py
# -*- coding: utf-8 -*-
"""
Single motion function with a dispatch by motion ID:
    platform_motion(model, data, platform_vel, motion_id)

motion_id:
  0 - static
  1 - sinusoidal motion along X axis
  2 - sinusoidal motion along Y axis
  3 - circular path in (X, Y)
  4 - smooth random walk (non-deterministic motion with varying speed)

`platform_vel` [m/s] sets the "effective" motion speed.
Call this function once per simulation step, before `mj_step()`.
"""

from __future__ import annotations
import math
import numpy as np
import mujoco

JOINT_X_NAME = "platform_x"
JOINT_Y_NAME = "platform_y"

# --- General geometric parameters ---
AMP_X   = 1.0   # amplitude along X [m] for motions 1 and 4 (soft bounds)
AMP_Y   = 1.0   # amplitude along Y [m] for motions 2 and 4 (soft bounds)
RADIUS  = 0.8   # radius for circular path [m] (motion 3)

# --- Internal state for motion 4 (smooth Random Walk) ---
_state = {
    "rw_last_t": None,
    "rw_pos": np.array([0.0, 0.0], dtype=float),
    "rw_vel": np.zeros(2, dtype=float),
    "seeded": False,
}

def reset_random_walk_state():
    """ Reset the internal state of the random-walk motion (use when restarting a simulation)."""
    _state["rw_last_t"] = None
    _state["rw_pos"].fill(0.0)
    _state["rw_vel"].fill(0.0)
    _state["seeded"] = False

# ---------- Utilities ----------
def _joint_addr(model, joint_name: str):
    j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if j_id == -1:
        raise ValueError(f"Joint '{joint_name}' not found in model.")
    return int(model.jnt_qposadr[j_id]), int(model.jnt_dofadr[j_id])

def _set_joint_state(data, qposadr: int, qveladr: int, q: float, qd: float):
    data.qpos[qposadr] = q
    data.qvel[qveladr] = qd

def _safe_dt(model, data) -> float:
    return max(model.opt.timestep, 1e-6)

# ---------- תנועות בסיס ----------
def _motion_static(model, data, platform_vel: float):
    # Do not modify qpos/qvel - the body remains where it is
    return

def _motion_x_sine(model, data, platform_vel: float):
    qpos_x, qvel_x = _joint_addr(model, JOINT_X_NAME)
    A = max(AMP_X, 1e-6)
    omega = abs(platform_vel) / A  # rad/s
    t = data.time
    x  = A * math.sin(omega * t)
    vx = A * omega * math.cos(omega * t) * (1 if platform_vel >= 0 else -1)
    _set_joint_state(data, qpos_x, qvel_x, x, vx)

def _motion_y_sine(model, data, platform_vel: float):
    qpos_y, qvel_y = _joint_addr(model, JOINT_Y_NAME)
    A = max(AMP_Y, 1e-6)
    omega = abs(platform_vel) / A
    t = data.time
    y  = A * math.sin(omega * t)
    vy = A * omega * math.cos(omega * t) * (1 if platform_vel >= 0 else -1)
    _set_joint_state(data, qpos_y, qvel_y, y, vy)

def _motion_circle(model, data, platform_vel: float):
    qpos_x, qvel_x = _joint_addr(model, JOINT_X_NAME)
    qpos_y, qvel_y = _joint_addr(model, JOINT_Y_NAME)
    R = max(RADIUS, 1e-6)
    direction = 1.0 if platform_vel >= 0 else -1.0
    omega = (abs(platform_vel) / R) * direction  # rad/s
    t = data.time
    cos_t = math.cos(omega * t)
    sin_t = math.sin(omega * t)
    x  = R * cos_t
    y  = R * sin_t
    vx = -R * omega * sin_t
    vy =  R * omega * cos_t
    _set_joint_state(data, qpos_x, qvel_x, x, vx)
    _set_joint_state(data, qpos_y, qvel_y, y, vy)

def _motion_random_walk(model, data, platform_vel: float):
    qpos_x, qvel_x = _joint_addr(model, JOINT_X_NAME)
    qpos_y, qvel_y = _joint_addr(model, JOINT_Y_NAME)

    if _state["rw_last_t"] is None:
        _state["rw_last_t"] = data.time
        _state["rw_pos"] = np.array([data.qpos[qpos_x], data.qpos[qpos_y]], dtype=float)
        _state["rw_vel"] = np.zeros(2, dtype=float)
        if not _state["seeded"]:
            np.random.seed(42)
            _state["seeded"] = True

    t  = data.time
    dt = max(t - _state["rw_last_t"], _safe_dt(model, data))
    _state["rw_last_t"] = t

    pos = _state["rw_pos"]
    vel = _state["rw_vel"]

    # Average target speed
    v_target = max(0.0, abs(platform_vel))

    # Soft noise acceleration + soft spring toward bounds
    noise_acc = 0.6 * np.random.randn(2)
    bounds = np.array([AMP_X, AMP_Y])
    soft_k = 0.6
    spring_acc = -soft_k * (pos / np.maximum(bounds, 1e-6))
    damping = 1.2

    acc = spring_acc + noise_acc - damping * vel
    vel = vel + acc * dt

    # Smooth normalization toward desired speed
    speed = float(np.linalg.norm(vel) + 1e-9)
    alpha = 0.15
    desired_speed = (1 - alpha) * speed + alpha * v_target
    vel *= desired_speed / speed

    pos = pos + vel * dt

    # Damped rebound at boundaries
    for i in (0, 1):
        limit = bounds[i]
        if abs(pos[i]) > limit:
            pos[i] = math.copysign(limit, pos[i])
            vel[i] *= -0.6

    _state["rw_pos"] = pos
    _state["rw_vel"] = vel

    _set_joint_state(data, qpos_x, qvel_x, float(pos[0]), float(vel[0]))
    _set_joint_state(data, qpos_y, qvel_y, float(pos[1]), float(vel[1]))

# ----------Unified API ----------
def platform_motion(model, data, platform_vel: float, motion_id: int):
    """
    Apply the motion law indicated by 'motion_id'.
    Call this once per step, before 'mj_step()'.
    """
    if motion_id == 0:
        return _motion_static(model, data, platform_vel)
    elif motion_id == 1:
        return _motion_x_sine(model, data, platform_vel)
    elif motion_id == 2:
        return _motion_y_sine(model, data, platform_vel)
    elif motion_id == 3:
        return _motion_circle(model, data, platform_vel)
    elif motion_id == 4:
        return _motion_random_walk(model, data, platform_vel)
    else:
        raise ValueError(f"Unknown motion_id: {motion_id} (expected 0..4)")
