# simulation_runner.py
"""
simulation_runner.py
====================
Minimal entry point to run **one** landing simulation via `orchestrator.run(...)`.

What this script does
---------------------
- Calls the high-level simulation orchestrator with a single configuration.
- Opens the interactive viewer (GLFW + MuJoCo) and runs until:
  * touchdown is detected, or
  * the time limit is reached, or
  * you close the window.

How to use
----------
Run directly from the command line:

    python simulation_runner.py

To change the scenario, edit the constants below (scene, motion, wind, noise, etc.)
and then re-run the script.

Parameters forwarded to `orchestrator.run(...)`
-----------------------------------------------
- scene_file : str
    Path to the MuJoCo XML scene (e.g., "scene.xml").
- time_limit : float
    Maximum simulated time in seconds.
- motion_id : int
    Platform motion mode (see `platform_motion.py`):
      0 = static,
      1 = sine along X,
      2 = sine along Y,
      3 = circle (X,Y),
      4 = smooth random-walk.
- platform_vel : float
    Nominal platform speed used by the chosen motion (m/s).
- wind_vel : float
    Constant horizontal wind speed (m/s). Set 0.0 for no wind.
- wind_direction : float
    Wind direction in **degrees** (0 = +X, 90 = +Y).
- fog : int
    1 to enable fog in the visualization; 0 to disable.
- sensor_noise : float
    Stddev of additive Gaussian noise added by `sensor_interface` (units match the signals).

Return value
------------
`orchestrator.run(...)` returns a boolean:
- True  → touchdown occurred (based on altitude threshold)
- False → no touchdown before termination

"""

from orchestrator import run

# -------- Configuration --------
SCENE_FILE: str      = "scene.xml"
TIME_LIMIT: float    = 600.0
MOTION_ID: int       = 1       # 0=static, 1=x-sine, 2=y-sine, 3=circle, 4=random-walk
PLATFORM_VEL: float  = 1.0     # m/s
WIND_VEL: float      = 0.0     # m/s (0.0 disables wind)
WIND_DIR_DEG: float  = 0.0     # degrees (0=+X, 90=+Y)
FOG: int             = 0       # 1=enable fog, 0=disable
SENSOR_NOISE: float  = 0.0     # standard deviation of Gaussian noise


if __name__ == "__main__":
    # Run the single scenario. The viewer will remain open until the sim ends or you close it.
    # The returned value is True if touchdown occurred, False otherwise.
    _landed: bool = run(
        scene_file=SCENE_FILE,
        time_limit=TIME_LIMIT,
        motion_id=MOTION_ID,
        platform_vel=PLATFORM_VEL,
        wind_vel=WIND_VEL,
        wind_direction=WIND_DIR_DEG,
        fog=FOG,
        sensor_noise=SENSOR_NOISE,
    )

