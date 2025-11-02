"""
drone_controller.py

Low_level controller that converts desired world_frame_velocity commands
into attitude (roll,pitch,yaw) and collective thrust, then computes
body torques with a PD attitude loop and writes controls via mujoco_data.

Pipeline (per step):
1) Velocity outer loop (P only here): a_des_world = kp_v * (v_cmd_world - v_world)
2) Gravity compensation & desired thrust direction:
      F_world = m * (a_des_world + g * e_z)
      z_body_des || F_world
3) Enforce desired yaw (hold current yaw if 'yaw_cmd' is None),
      build desired rotation R_des (body->world), extract (φ, θ, ψ).
4) Attitude PD on SO(3) to compute body torques τ.
5) Send [thrust, τx, τy, τz] to actuators.

Frames & Units
- Velocities: world frame [vx,vy,vz] in m/s
- Acceleration: world frame in m/s^2
- Attitude: roll φ, pitch θ and yaw ψ in rad
- Torques: body frame [mx,my,mz] in N*m
- Thrust: collective in Newtons (acts along +body-z)
"""

from __future__ import annotations
import math
import numpy as np
import mujoco
import mujoco_data as md

kp_v = np.array([-2, 2, 2.4])
kd_v = np.array([0.6, 0.6, 0.0])
kp_att = np.array([6.0, 6.0, 3.0])
kd_att = np.array([0.15, 0.15, 0.08])
MAX_TILT_DEG = 25.0

def yaw_from_Rbw(Rbw):
    """
     Extract world yaw ψ from a body→world rotation matrix.

     Parameters
     ----------
     Rbw : (3,3) array-like
         Rotation from body frame to world frame.

     Returns
     -------
     float
         Yaw angle ψ in radians, using atan2(R[1,0], R[0,0]).
     """
    return math.atan2(Rbw[1,0], Rbw[0,0])

def desired_attitude_and_thrust(mass, g_pos, a_des_world, Rbw_now, yaw_cmd):
    """
        Map desired world acceleration to desired attitude (φ, θ, ψ) and thrust.

        Steps
        -----
        - Compute desired total force F_world = m * (a_des + g * e_z).
        - Desired body +Z axis aligns with F_world (thrust direction).
        - Honor requested yaw (`yaw_cmd`); if None, keep current yaw.
        - Build R_des with columns [x_d, y_d, z_b], then extract roll & pitch.
        - Limit roll/pitch by MAX_TILT_DEG; thrust magnitude = ||F_world||.

        Parameters
        ----------
        mass : float
            Vehicle mass (kg).
        g_pos : float
            Gravity magnitude (m/s²), e.g., 9.81.
        a_des_world : (3,) array-like
            Desired world-frame acceleration (m/s²) from outer loop.
        Rbw_now : (3,3) array-like
            Current body→world rotation.
        yaw_cmd : float | None
            Desired yaw (rad). If None, keep current yaw.

        Returns
        -------
        (phi, theta, yaw_des, Tdes) : tuple[float, float, float, float]
            Roll (rad), pitch (rad), yaw (rad), and thrust (N).
        """
    e3 = np.array([0.0, 0.0, 1.0])
    Fw = mass * (a_des_world + g_pos * e3)  #desired total force in world
    Fn = float(np.linalg.norm(Fw) + 1e-9)   #magnitude (avoid divide by zero)
    zb = Fw / Fn                            # desired body +Z axis (world)

    # Desired yaw: keep current if none provided
    yaw_des = yaw_from_Rbw(Rbw_now) if yaw_cmd is None else float(yaw_cmd)

    # Construct desired orientation:
    # First choose an x-axis consistent with yaw, then make y = zb × x, x = y × zb
    xc = np.array([math.cos(yaw_des), math.sin(yaw_des), 0.0])
    yd = np.cross(zb, xc); yd /= (np.linalg.norm(yd) + 1e-12)
    xd = np.cross(yd, zb); xd /= (np.linalg.norm(xd) + 1e-12)
    Rdes = np.column_stack((xd, yd, zb))    # body axes expressed in world

    # Extract roll/pitch from R_des
    phi = math.atan2(Rdes[2,1], Rdes[2,2])
    theta = math.asin(-Rdes[2,0])

    # Enforce tilt limits
    lim = math.radians(MAX_TILT_DEG)
    phi = float(max(-lim,min(lim, phi)))
    theta = float(max(-lim,min(lim,theta)))

    Tdes = Fn        # thrust magnitude in Newtons
    return phi, theta, yaw_des, Tdes

def attitude_pd_torque(Rbw_now, omega_body, phi_des, theta_des, yaw_des):
    """
        Attitude PD on SO(3): compute body torques τ = -Kp*e_R - Kd*ω.

        Implementation details
        ----------------------
        - Rebuild R_des from Euler targets (Z-Y-X order): Rz(yaw)*Ry(pitch)*Rx(roll).
        - Compute skew-symmetric error: E = R_desᵀ R_now − R_nowᵀ R_des.
        - vee(E) gives 3-vector attitude error e_R.
        - Apply per-axis PD gains in body coordinates.

        Parameters
        ----------
        Rbw_now : (3,3) array-like
            Current body→world rotation.
        omega_body : (3,) array-like
            Body angular rates [p,q,r] in rad/s.
        phi_des, theta_des, yaw_des : float
            Desired roll, pitch, yaw (rad).

        Returns
        -------
        np.ndarray shape (3,)
            Body torques [mx, my, mz] in N·m.
        """
    cz, sz = math.cos(yaw_des), math.sin(yaw_des)
    cy, sy = math.cos(theta_des), math.sin(theta_des)
    cx, sx = math.cos(phi_des), math.sin(phi_des)
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    Ry = np.array([[cy,0,-sy],[0,1,0],[sy,0,cy]])
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    Rdes = Rz @ Ry @ Rx

    E = Rdes.T @ Rbw_now - Rbw_now.T @ Rdes
    eR = 0.5 * np.array([E[2,1], E[0,2],E[1,0]])

    tau = -kp_att * eR - kd_att * omega_body
    return tau

def execute_command(model,data,v_cmd_world,v_world,Rbw,omega_body,mass,g_pos,yaw_cmd):
    """
        Main entry: convert desired world velocity into attitude/thrust and write actuators.

        Parameters
        ----------
        model, data : mujoco.MjModel, mujoco.MjData
            MuJoCo handles.
        v_cmd_world : (3,) array-like
            Desired world-frame velocity [vx, vy, vz] in m/s.
        v_world : (3,) array-like
            Measured world-frame velocity [vx, vy, vz] in m/s.
        Rbw : (3,3) array-like
            Body→world rotation.
        omega_body : (3,) array-like
            Body angular rates [p, q, r] in rad/s.
        mass : float
            Vehicle mass (kg).
        g_pos : float
            Gravity magnitude (m/s²), e.g., 9.81.
        yaw_cmd : float | None
            Desired yaw (rad). If None, keep current yaw.

        Returns
        -------
        None
            Writes (thrust, mx, my, mz) via `md.set_actuator_control`.
        """
    v_cmd_world = np.asarray(v_cmd_world, dtype = float)
    v_world = np.asarray(v_world, dtype = float)
    #Outer-loop: P on velocity (per-axis)
    a_des_world = kp_v * (v_cmd_world - v_world)

    # Map to desired attitude & thrust (gravity compensated)
    phi_des, theta_des, yaw_des, T_des = desired_attitude_and_thrust(mass,g_pos, a_des_world, Rbw, yaw_cmd)

    #Inner-loop: attitude PD -> body torques
    tau = attitude_pd_torque(Rbw, np.asarray(omega_body, dtype = float), phi_des, theta_des, yaw_des)

    #Write actuators (clip thrust to be non-negative)
    md.set_actuator_control(model, data, thrust_N = float(max(0.0, T_des)), mx = float(tau[0]), my = float(tau[1]), mz = float(tau[2]))
