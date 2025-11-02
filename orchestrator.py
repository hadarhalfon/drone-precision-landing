"""
orchestrator.py
===============
High-level simulation loop tying together model load, visualization, sensing,
decision-making, low-level control, platform motion, and simple landing checks.

Execution flow (per run)
------------------------
1) Load model & create data
2) Initialize visualization (window, overlays, fog)
3) Configure wind (if requested) and reset platform random walk (if used)
4) Main loop:
   - Read noisy sensor measurements (drone & platform pos/vel)
   - Compute relative errors and altitude-to-land
   - Build body/world kinematics (Rbw, omega_body)
   - Call high-level landing algorithm `make_decision(...)`
   - Build world velocity command and pass to low-level controller
   - Step platform motion and the physics
   - Update overlay text and render
   - Early-exit in two cases:
        * heuristic says landing is not feasible (after t>10s)
        * altitude threshold indicates touchdown
5) If landed, record metrics (distance/time/vertical speed)
6) Show a final status message (based on feasibility code) until the window closes

Notes
-----.
- Environment variable `MUJOCO_GL` is set to "glfw" by default to ensure an
  interactive windowed context.
"""
import mujoco
import visualization as viz
import platform_motion as pm
import wind
import sensor_interface as si
import numpy as np
import calculator
import landing_evaluation as le
from mujoco_data import  landing_possibility, print_after_end
import os

from mujoco_model import load_model
import mujoco_data  as md
from landing_algorithm import make_decision
from drone_controller import execute_command

#Use GLFW rendering unless the user overrides externally.
os.environ.setdefault("MUJOCO_GL","glfw")


def run(scene_file,time_limit,motion_id,platform_vel,wind_vel,wind_direction,fog,sensor_noise):
    """
       Run one interactive simulation until time limit, window close, or touchdown.

       Parameters
       ----------
       scene_file : str
           Path to the MuJoCo XML scene file.
       time_limit : float
           Maximum simulated time (seconds).
       motion_id : int
           Platform motion mode: 0=static, 1=x, 2=y, 3=circular, 4=random-walk.
       platform_vel : float
           Nominal platform speed parameter for the chosen motion (m/s).
       wind_vel : float
           Constant wind speed (m/s). Set 0 for no wind.
       wind_direction : float
           Wind direction in degrees (0 = +X).
       fog : int
           1 to enable fog in the visualization, else 0.
       sensor_noise : float
           Stddev of additive Gaussian noise for sensor readings.

       Returns
       -------
       bool
           True if touchdown was detected (based on altitude threshold), else False.
       """
    landed = False

    # --- Model & Visualization setup ---
    model = load_model(scene_file)
    data = md.make_data(model)
    mujoco.mj_forward(model,data)
    viz.init(model,data,window_title="Drone Landing on Moving Platform")
    viz.set_status_text(None)
    viz.set_overlay_data()  #Initializes any internal overlay state in the viz module

    if fog == 1:
        viz.enable_fog(model)

    #--- Wind configuration (constant field) ---
    if float(wind_vel) != 0.0:
        wind.ensure_fluid_enabled(model)
        wind.set_wind_constant(model,speed_mps=(float(wind_vel)),direction_deg=wind_direction)

    # Reset internal state for random-walk platform motion if selected
    if motion_id == 4:
        pm.reset_random_walk_state()

    # Basic physical properties (used by the low-level controller)
    drone_bid = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"cf2")
    mass = float(model.body_mass[drone_bid])
    g_pos = float(-model.opt.gravity[2])   # 9.81 if +Z is up
    # Heuristic feasibility check (used for early abort & final message)
    can_land = landing_possibility(motion_id,platform_vel,sensor_noise,wind_vel)
    # Simulation timing and counters
    max_steps = int(time_limit/model.opt.timestep)
    step = 0

    # --- Main loop ---
    while step<max_steps and not viz.should_close():

        # 1) Sense with optional noise
        d_pos = si.read_drone_position(model,data,sensor_noise)
        p_pos = si.read_platform_position(model,data,sensor_noise)
        d_vel = si.read_drone_velocity(model,data,sensor_noise)
        p_vel = si.read_platform_velocity(model,data,sensor_noise)

        # 2) Relative geometry & kinematics
        dist = calculator.calculate_distance(d_pos,p_pos)
        dist_x = calculator.calculate_distance_x(d_pos,p_pos)
        dist_y = calculator.calculate_distance_y(d_pos,p_pos)
        altitude = calculator.calculate_altitude(d_pos,p_pos)
        rel_vel_x = calculator.calculate_relative_velocity_x(d_vel,p_vel)
        rel_vel_y = calculator.calculate_relative_velocity_y(d_vel,p_vel)

        # MuJoCo packs angular (first 3) then linear (last 3) when using mj_objectVelocity
        omega_world = d_vel[:3]
        v_world = d_vel[3:]
        p_vel_world = p_vel[3:]

        # Rotation & body rates
        Rbw = np.array(data.xmat[drone_bid], dtype=float).reshape(3, 3)
        omega_body = Rbw.T @ omega_world

        sim_time = data.time

        # 3) High-level decision (landing algorithm)
        speed_x,speed_y,speed_z = make_decision(dist,dist_x,dist_y,altitude,rel_vel_x,rel_vel_y,p_vel_world[0],p_vel_world[1])
        v_cmd_world = np.array([speed_x + p_vel_world[0], speed_y + p_vel_world[1],speed_z], dtype = float)

        # 4) Low-level control (attitude/thrust + torque write)
        execute_command(model, data, v_cmd_world, v_world, Rbw, omega_body, mass, g_pos, yaw_cmd=None)

        # 5) Move platform according to the selected motion law, then step physics
        pm.platform_motion(model, data, platform_vel, motion_id) #platform_motion: 0-static, 1-x, 2-y, 3-circular, 4-random
        mujoco.mj_forward(model,data) # ensure derived quantities are consistent
        md.step_simulation(model,data)
        step+=1

        # Early abort if the heuristic says landing is impossible (after 10s)
        if data.time > 10.0:
            if can_land != 0:
                break

        # Touchdown check by altitude threshold
        if altitude <= 0.01:
            landed = True
            break

        # 6) Overlay text and draw a frame
        note=(f"t={data.time:5.2f}s | dxy={dist:0.3f}"
              f"| dx={dist_x:0.3f} dy={dist_y:0.3f} | alt={altitude:0.3f}"
              f"| vrel=({rel_vel_x:0.2f},{rel_vel_y:0.2f})"
              f"| {'LANDED' if landed else 'FLYING'}")

        viz.set_overlay_data(note=note)
        viz.render_once()

    # --- Record metrics on touchdown ---
    if landed:
        le.evaluate_distance(dist)
        le.evaluate_time(sim_time)
        le.evaluate_velocity(d_vel)

    # --- Final status message (based on feasibility code)---
    str_to_print = print_after_end(can_land)

    while not viz.should_close():
        viz.set_status_text(str_to_print)
        viz.render_once()

    viz.cleanup()
    return landed







