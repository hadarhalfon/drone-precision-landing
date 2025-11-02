# Drone Landing on a Moving Platform (MuJoCo)

An end-to-end simulation of a quadrotor that aligns over — and lands on — a (possibly moving) platform using MuJoCo.  
Includes a landing algorithm, PD attitude controller, platform motion models, wind, optional sensor noise, evaluation (CSV + plots), and an interactive viewer.

---

## Quick Start

1. **Install dependencies**
   ```bash
   pip install mujoco glfw numpy matplotlib
   ```
   Make sure you have a working OpenGL driver.

2. **Provide a MuJoCo XML scene**
   Place `scene.xml` in the repo root (or adjust the path in `simulations.py`).

3. **Run the batch scenarios**
   ```bash
   python simulations.py
   ```
   This opens the viewer, runs all scenarios, and writes:
   - CSV: `results/landing_results.csv` (with a `# SUMMARY` section)
   - Plots: `results/plots/` (distance [cm], time [s], vertical speed [+down, m/s])

Run a single scenario from Python:
```python
from orchestrator import run
run(scene_file="scene.xml", time_limit=600.0,
    motion_id=3, platform_vel=0.7,
    wind_vel=1.02, wind_direction=90.0,
    fog=0, sensor_noise=0.0)
```

---

## Viewer Controls

- **ESC**: close window  
- **F**: toggle follow camera ON/OFF  
- **Arrow Keys**: orbit (azimuth/elevation)  
- **- / =**: zoom out / in  
- **W/A/S/D**: pan in view plane  
- **Q / E**: pan down / up

Bottom-right: one-line telemetry overlay.  
Top-left: status banner (e.g., landing result).

---

## Modules

- `orchestrator.py` — main simulation loop (model load, wind, platform motion, sensing, control, stepping, rendering, logging).
- `mujoco_model.py` — loads and compiles the MuJoCo XML.
- `mujoco_data.py` — MuJoCo helpers: resolve IDs, read world poses/vels, write actuator controls, clamp.
- `sensor_interface.py` — reads states via `mujoco_data` and optionally adds Gaussian noise.
- `landing_algorithm.py` — high-level velocity landing logic (CHASE → ALIGN → DESCEND) with descent gates.
- `drone_controller.py` — world-velocity command → desired attitude + thrust → PD attitude torques → actuator outputs.
- `platform_motion.py` — platform kinematics: static, sine (X/Y), circle, smooth random-walk.
- `wind.py` (or `env_effects.py`) — constant wind + fluid properties (used by `orchestrator.run(...)`).
- `calculator.py` — light math helpers (distance, altitude-to-land, relative velocities).
- `landing_evaluation.py` — collects per-landing metrics, prints summary, saves CSV, plots histograms.
- `visualization.py` — GLFW + mjv/mjr viewer with follow camera and consolidated overlay.
- `simulations.py` — predefined scenarios and a batch runner.

---

## How It Works (Flow)

```
sensor_interface → calculator → landing_algorithm.make_decision()
      → drone_controller.execute_command() → mujoco_data.set_actuator_control()
      → platform_motion + wind → visualization → landing_evaluation
```

- **Command type**: desired **world-frame** linear velocities `[vx, vy, vz]`.
- **Touchdown vertical speed** is stored as **positive downward** (i.e., `-vz` if +Z is up).

---

## Coordinates & Units

- **World frame**: meters `[x, y, z]`, **+Z is up**
- **Velocities**: linear m/s, angular rad/s
- **Angles**: radians unless stated otherwise
- **Wind direction**: degrees (0° = +X, 90° = +Y)

Model-specific Z offsets used in altitude calculation (edit in `calculator.py` to match your XML):
- `DRONE_BOTTOM_FROM_ORIGIN_Z`
- `PLATFORM_TOP_FROM_ORIGIN_Z`

---

## Platform Motions

Call once per step (done for you in `orchestrator.py`):

| `motion_id` | Motion                     | Key params            |
|-------------|----------------------------|-----------------------|
| 0           | Static                     | —                     |
| 1           | Sine along **X**           | `AMP_X`               |
| 2           | Sine along **Y**           | `AMP_Y`               |
| 3           | Circle in (X,Y)            | `RADIUS`              |
| 4           | Smooth random-walk (X,Y)   | `AMP_X`, `AMP_Y`      |

`platform_vel` (m/s) sets the effective motion speed.

---

## Wind 

Wind is configured **inside `orchestrator.run(...)`** whenever `wind_vel` (m/s) is non‑zero.  
Set `wind_vel` and `wind_direction` in your scenario (in `simulations.py`) or pass them to `run(...)` directly (see the Quick Start example above).

---

## Tuning Knobs

- `landing_algorithm.py`: XY chase/hold gains, alignment thresholds, descent gates, flare limits.
- `drone_controller.py`: velocity P gains, attitude PD gains, max tilt (deg).
- `platform_motion.py`: amplitudes/radius and random-walk shaping.
- `sensor_interface.py`: noise std (`sensor_noise` argument).

---

## XML Names Expected

If your XML uses different names, update them in code:

- **Bodies**: `"cf2"` (drone), `"platform"`
- **Platform joints**: `"platform_x"`, `"platform_y"`
- **Actuators**: `"u_thrust"`, `"u_mx"`, `"u_my"`, `"u_mz"`

---

## Outputs

- **Console**: per-run status + final summary
- **CSV**: `results/landing_results.csv` (per-landing rows + `# SUMMARY`)
- **Plots**: PNG histograms in `results/plots/` (distance [cm], time [s], vertical speed [+down, m/s])

---

## Troubleshooting

- **GLFW / window issues**: ensure OpenGL drivers; `pip show glfw` to confirm install.
- **No wind effect**: ensure your scene has positive fluid properties or let `orchestrator.run(...)` call the wind helper.
- **Altitude seems wrong**: update the Z offsets in `calculator.py`.

---

## License

[MIT]


