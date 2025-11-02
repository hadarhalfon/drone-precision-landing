"""
simulations.py
==============
Batch runner for the landing experiments.

What this module provides
-------------------------
- `Simulation` dataclass capturing the knobs for a single run.
- A predefined `simulations` list with varied scenarios (motion, wind, fog, noise).
- `run_all(simulations)` which:
    * calls `orchestrator.run(**asdict(cfg))` for each scenario,
    * tracks how many runs ended with a landing,
    * prints a summary using `landing_evaluation` helpers,
    * saves a CSV and plots histograms.

Notes
-----
- This file **does not** change control/physics logic; it only sequences runs,
  gathers results, and produces artifacts.
- Printed status messages.
"""
from dataclasses import dataclass, asdict
from typing import List
from orchestrator import run
from landing_evaluation import compute_summary, print_summary, save_csv, plot_histograms

@dataclass()
class Simulation:
    """
    Configuration for a single simulation run.

    Fields
    ------
    scene_file : str
        MuJoCo XML file to load (default: "scene.xml").
    time_limit : float
        Maximum simulated time in seconds (default: 600.0).
    motion_id : int
        Platform motion mode: 0=static, 1=x-sine, 2=y-sine, 3=circle, 4=random-walk.
    platform_vel : float
        Nominal platform speed [m/s] used by the selected motion.
    wind_vel : float
        Constant wind magnitude [m/s] (0 disables wind).
    wind_direction : float
        Wind direction in degrees (0 = +X).
    fog : int
        1 to enable fog in visualization, else 0.
    sensor_noise : float
        Stddev of additive Gaussian noise applied by the sensor interface.
    """
    scene_file: str = "scene.xml"
    time_limit: float = 600.0
    motion_id: int = 0
    platform_vel: float = 0.0
    wind_vel: float = 0.0
    wind_direction: float = 0.0
    fog: int = 0
    sensor_noise: float = 0.0


# A set of scenarios covering different motions, winds, fog, and noise levels.
simulations: List[Simulation] = [
    Simulation(motion_id = 0),
    Simulation(motion_id = 0, wind_vel = 1.54, wind_direction = 90.0),
    Simulation(motion_id = 0, fog = 1),
    Simulation(motion_id = 0, sensor_noise = 0.03),
    Simulation(motion_id = 1, platform_vel = 1.0),
    Simulation(motion_id = 2, platform_vel = 0.3),
    Simulation(motion_id = 1, platform_vel = 1.0, wind_vel = 1.02, wind_direction = 90.0),
    Simulation(motion_id = 2, platform_vel = 1.0, wind_vel = 1.02, wind_direction = 90.0),
    Simulation(motion_id = 2,platform_vel = 0.5, fog = 1),
    Simulation(motion_id = 1,platform_vel = 0.5, fog = 1),
    Simulation(motion_id = 2, platform_vel = 0.7, sensor_noise = 0.03),
    Simulation(motion_id = 3, platform_vel = 0.4),
    Simulation(motion_id = 3, platform_vel = 0.7),
    Simulation(motion_id = 3, platform_vel = 0.7, wind_vel = 1.02, wind_direction = 90.0),
    Simulation(motion_id = 3, platform_vel = 0.8, wind_vel = 1.02, wind_direction = 90.0),
    Simulation(motion_id = 3, platform_vel = 0.8, wind_vel = 0.51, wind_direction = 90.0),
    Simulation(motion_id = 3, platform_vel = 0.6, fog = 1),
    Simulation(motion_id = 3, platform_vel = 0.7, sensor_noise = 0.06),
    Simulation(motion_id = 4, platform_vel = 0.5),
    Simulation(motion_id = 4, platform_vel = 0.6),
    Simulation(motion_id = 4, platform_vel = 0.1),
    Simulation(motion_id = 4, platform_vel = 0.8),
    Simulation(motion_id = 4, platform_vel = 0.1, wind_vel = 1.02, wind_direction = 90.0),
    Simulation(motion_id = 4, platform_vel = 0.4, fog = 1),
    Simulation(motion_id = 4, platform_vel = 0.3, sensor_noise = 0.04)
]




def run_all(simulations: List[Simulation]) -> int:
    """
    Run all simulations, report per-run status, then print and save a summary.

    Parameters
    ----------
    simulations : list[Simulation]
        Scenarios to execute in sequence.

    Returns
    -------
    int
        Number of runs that ended with a landing.
    """
    landed_count = 0
    total = len(simulations)

    for i, cfg in enumerate(simulations, start=1):
        print(f"\n[run_all] simulation {i}/{total}: {cfg}")
        try:
            landed: bool = run(**asdict(cfg))
        except Exception as e:
            print(f"[run_all] ❌ error in simulation {i}: {e}")
            landed = False

        if landed:
            landed_count += 1
            print(f"[run_all] ✅ successful landing (total : {landed_count})")
        else:
            print(f"[run_all] ⛔ not landed")

    print(f"\n[run_all] conclusion: {landed_count}/{total} simulations ended with landing.")
    summary = compute_summary(total_runs=total)
    print_summary(summary)

    csv_path = save_csv("results/landing_results.csv", extra_meta={"total_runs": total})
    print(f"[run_all] CSV saved to: {csv_path}")

    plot_paths = plot_histograms(save_dir="results/plots", show=True)
    if plot_paths:
        for p in plot_paths:
            print(f"[run_all] Plot saved: {p}")

if __name__ == "__main__":
    run_all(simulations)
