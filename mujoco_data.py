"""
mujoco_data.py
==============
Thin helpers around MuJoCo `MjModel` / `MjData` for:
- Resolving body/actuator IDs by name (with one-time caching for actuators)
- Reading world positions and velocities for named bodies
- Writing actuator commands (collective thrust + body torques)
- Simple feasibility/diagnostic helpers (`landing_possibility`, `print_after_end`)

Conventions
-----------
- Positions returned by `get_position` are world-frame `[x, y, z]` (meters).
- Velocities returned by `get_velocity` are a 6-vector:
    [ωx, ωy, ωz, vx, vy, vz] (angular first, then linear).
- Actuator names must exist in the XML; they are resolved once and cached
  in `ACT_IDS` on the first call to `set_actuator_control`.
"""

import mujoco
import numpy as np

# ---- Body and actuator names (must match the xml file) ----
DRONE_NAME = "cf2"
PLATFORM_NAME = "platform"
DISK_NAME = "success_zone_disk"
ACT_NAMES = ("u_thrust", "u_mx", "u_my", "u_mz")

# Cache for actuator IDs (resolved once per process)
ACT_IDS = None



def make_data(model):
    """
    Create a fresh 'MjData' for the given 'MjModel'.
    """
    return mujoco.MjData(model)

def body_id(model,body_name):
    """
        Resolve a MuJoCo body name to its numeric id.

        Raises
        ------
        ValueError
            If the body name is not found in the model.
        """
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid == -1:
        raise ValueError(f"Body '{body_name}' not found in model.")
    return int(bid)

def get_position(model, data,bid):
    """
    World position of body 'bid' as a float array '[x,y,z]'.
    """
    return np.array(data.xpos[bid],dtype=float)

def get_velocity(model, data, bid):
    """
        6D body velocity for `bid`: `[wx, wy, wz, vx, vy, vz]`.

        Primary method uses `mj_objectVelocity` (world-frame twist).
        If unavailable (older MuJoCo), falls back to `data.cvel` (spatial body
        velocity in body frame) and rotates into world frame using `xmat`.
        """
    out = np.zeros(6,dtype = float)
    try:
        mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,bid,out,0)
        return out.copy()
    except Exception:
        cvel = np.array(data.cvel[bid], dtype = float)  # [ang(body), lin(body)]
        w_b, v_b = cvel[:3], cvel[3:]
        R = np.array(data.xmat[bid],dtype = float).reshape(3,3)  #body -> world
        w_w = R @ w_b
        v_w = R @ v_b
        return np.concatenate([w_w,v_w], axis = 0)

def get_drone_position(model,data):
    """
    World position of the drone body ('DRONE_NAME').
    """
    bid = body_id(model, DRONE_NAME)
    return get_position(model,data,bid)

def get_platform_position(model,data):
    """
    World position of the platform body ('PLATFORM_NAME').
    """
    bid = body_id(model, PLATFORM_NAME)
    return get_position(model, data, bid)

def get_drone_velocity(model,data):
    """
    6D velocity '[wx, wy, wz, vx, vy, vz]' of the drone body.
    """
    bid = body_id(model, DRONE_NAME)
    return get_velocity(model,data,bid)

def get_platform_velocity(model,data):
    """
    6D velocity '[wx, wy, wz, vx, vy, vz]' of the platform body.
    """
    bid = body_id(model,PLATFORM_NAME)
    return get_velocity(model,data,bid)

def resolve_actuators_once(model):
    """
    Resolve actuator IDs from 'ACT_NAMES' once and cache them in 'ACT_IDS'.

    Raises:
         RuntimeError - if any actuator name is not found in the model.
    """
    global ACT_IDS
    if ACT_IDS is not None:
        return
    ids = []
    for nm in ACT_NAMES:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, nm)
        if aid == -1:
            # (Keeping the original message text; adjust if desired)
            raise RuntimeError(f"Actuator '{nm}' not found in mode.")
        ids.append(int(aid))
    ACT_IDS = tuple(ids)

def clamp(model,aid, u):
    """
    Clamp a control value 'u' into the actuator's control range.
    """
    lo, hi = model.actuator_ctrlrange[aid]
    return float(max(lo,min(hi,u)))


def set_actuator_control(model, data, thrust_N, mx, my, mz):
    """
    Write actuator controls: collective thrust and body torques.

    Parameters
    float: thrust_N: collective thrust in Newtons.
    float: mx, my, mz: Body torques (N·m) about x, y, z respectively
    """
    resolve_actuators_once(model)
    aT, aMx, aMy, aMz = ACT_IDS
    data.ctrl[aT] = clamp(model, aT, float(thrust_N))
    data.ctrl[aMx] = clamp(model, aMx , float(mx))
    data.ctrl[aMy] = clamp(model, aMy, float(my))
    data.ctrl[aMz] = clamp(model, aMz, float(mz))

def step_simulation(model,data):
    """
    Advance the simulation by one step.
    """
    mujoco.mj_step(model,data)

def landing_possibility(motion_id,platform_vel,_sensor_noise,wind_vel):
    """
    Heuristic feasibility code for landing, based on motion, speeds, and wind.

    :returns
    int
        0: OK (possible)
        1: Platform velocity too high for the chosen motion
        2: Sensor noise present
        3: Wind too strong (thresholds depend on motion)
    """
    if ((motion_id == 1 or motion_id == 2) and platform_vel > 1.0) or (motion_id == 3 and (platform_vel > 0.9 or platform_vel < 0.5)) or (motion_id == 4 and platform_vel > 0.6):
        return 1
    if _sensor_noise > 0.0:
        return 2
    if (motion_id == 0 and wind_vel > 1.54) or (motion_id != 0 and wind_vel > 1.02) or (motion_id == 3 and platform_vel > 0.7 and wind_vel > 0.51):
        return 3
    return 0

def print_after_end(num):
    """
    Human-readable message for a result/feasibility code.

    Codes
        0 -> success
        1 -> platform velocity too high
        2 -> sensor noise
        3 -> wind too strong
    """
    if num == 0:
        return "The drone has successfully landed!"
    if num == 1:
        return  "Can't land, the platform velocity is too high."
    if num == 2:
        return "Can't land, sensor noise."
    if num == 3:
        return "can't land, wind is too strong."