"""
landing_evaluation.py
=====================
Collect and summarize per-landing metrics during simulations.

Tracked metrics
---------------
- DISTANCES : horizontal distance from the platform center at touchdown (meters)
- TIMES     : touchdown time since the start of the simulation (seconds)
- VEL_Z     : vertical speed at touchdown (m/s), stored as **+downward**

Notes
-----
- Storage units are SI (m, s, m/s). When exporting/plotting we may display
  distance in centimeters for convenience.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Tuple, Iterable
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from calculator import as_xyz

# Per-landing metric buffers (append-only during a batch run)
DISTANCES: list[float] = [] # horizontal distance from platform center [m]
TIMES: list[float] = [] # touchdown time since start of run [s]
VEL_Z: list[float] = [] # vertical speed at touchdown [m/s], positive downward

def evaluate_distance(dist):
    """
        Record horizontal distance from center at touchdown.

        Parameters
        ----------
        dist : float
            Distance in meters.
        """
    DISTANCES.append(float(dist))

def evaluate_time(t_seconds):
    """
       Record touchdown time since the start of the simulation.

       Parameters
       ----------
       t_seconds : float
           Time in seconds.
       """
    TIMES.append(float(t_seconds))

def evaluate_velocity(d_vel):
    """
        Record vertical touchdown speed, stored as **positive downward**.

        Parameters
        ----------
        d_vel : array-like
            Any vector `as_xyz` can parse; vz is taken from the z-component.
            If +Z is up (MuJoCo default), downward speed is -vz.
        """
    v = as_xyz(d_vel)
    vz = float(v[2])
    VEL_Z.append(-vz) # store as +downward

# ===== Statistics & summary =====
@dataclass(frozen=True)
class Summary:
    total_runs: int
    landed_runs: int
    success_rate: float
    # Display units
    mean_distance: Optional[float]      # centimeters (cm)
    median_distance: Optional[float]    # centimeters (cm)
    std_distance: Optional[float]       # centimeters (cm)
    mean_time: Optional[float]          # seconds (s)
    median_time: Optional[float]        # seconds (s)
    std_time: Optional[float]           # seconds (s)
    mean_vel_z: Optional[float]         # meters/seconds (m/s)
    median_vel_z: Optional[float]       # meters/seconds (m/s)
    std_vel_z: Optional[float]          # meters/seconds (m/s)

def _safe_stats(arr: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
        Return (mean, median, std) for a list, or (None, None, None) if empty.
        """
    if not arr:
        return None, None, None
    a = np.asarray(arr, dtype=float)
    return float(a.mean()), float(np.median(a)), float(a.std(ddof=1) if a.size > 1 else 0.0)

def compute_summary(total_runs: int) -> Summary:
    """
        Compute aggregate statistics for the recorded landings.

        Parameters
        ----------
        total_runs : int
            Number of simulation attempts in the batch (landed or not).

        Returns
        -------
        Summary
            Data class with success rate and basic stats for each metric.
        """
    landed_runs = len(DISTANCES)
    success_rate = (landed_runs / total_runs) if total_runs > 0 else 0.0

    # Distances are stored in meters; present them in centimeters
    distances_cm = [d * 100.0 for d in DISTANCES]
    m_d, md_d, sd_d = _safe_stats(distances_cm)

    # Time in seconds; vertical speed in m/s (stored +downward)
    m_t, md_t, sd_t = _safe_stats(TIMES)
    m_v, md_v, sd_v = _safe_stats(VEL_Z)

    return Summary(
        total_runs=total_runs,
        landed_runs=landed_runs,
        success_rate=success_rate,
        mean_distance=m_d, median_distance=md_d, std_distance=sd_d,
        mean_time=m_t, median_time=md_t, std_time=sd_t,
        mean_vel_z=m_v, median_vel_z=md_v, std_vel_z=sd_v,
    )

def print_summary(summary: Summary) -> None:
    """
        Pretty-print the summary to stdout.
        """
    print("\n======== Landing Summary ========")
    print(f"Total runs:          {summary.total_runs}")
    print(f"Landed runs:         {summary.landed_runs}")
    print(f"Success rate:        {summary.success_rate:.1%}")
    if summary.mean_distance is not None:
        print("\n-- Distance from center at touchdown (cm) --")
        print(f"mean={summary.mean_distance:.3f}  median={summary.median_distance:.3f}  std={summary.std_distance:.3f}")
        print("\n-- Touchdown time (s) --")
        print(f"mean={summary.mean_time:.3f}  median={summary.median_time:.3f}  std={summary.std_time:.3f}")
        print("\n-- Vertical speed at touchdown (+down, m/s) --")
        print(f"mean={summary.mean_vel_z:.3f}  median={summary.median_vel_z:.3f}  std={summary.std_vel_z:.3f}")
    else:
        print("\nNo successful landings → no per-landing stats.")

def save_csv(path: str, extra_meta: Optional[Dict[str, Any]] = None) -> str:
    """
     Create a CSV with per-landing rows and an appended summary.

     Columns:
       distance_cm (cm), time_s (s), vel_z_mps (m/s, positive downward)

     The file may begin with key/value metadata rows prefixed by '# '.

     Returns
     -------
     str
         Absolute path to the written CSV.
     """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        if extra_meta:
            for k, v in extra_meta.items():
                w.writerow([f"# {k}", v])

        # Header with explicit units
        w.writerow(["idx", "distance_cm", "time_s", "vel_z_mps"])

        # Landing rows (distance converted to cm here)
        for i, (d_m, t_s, vz_mps) in enumerate(zip(DISTANCES, TIMES, VEL_Z), start=1):
            w.writerow([i, f"{d_m * 100.0:.6f}", f"{t_s:.6f}", f"{vz_mps:.6f}"])

        # Summary section
        summary = compute_summary(total_runs=extra_meta.get("total_runs", len(DISTANCES)) if extra_meta else len(DISTANCES))
        w.writerow([])
        w.writerow(["# SUMMARY"])
        for k, v in asdict(summary).items():
            w.writerow([k, v])
    return os.path.abspath(path)

def plot_histograms(save_dir: Optional[str] = None, show: bool = True) -> List[str]:
    """
    Plot three histograms: distance (cm), time (s), vertical speed (m/s).

    Parameters
    ----------
    save_dir : str | None
        If provided, PNGs are saved there and absolute paths are returned.
    show : bool
        If True, display the figures; otherwise close them after saving.

    Returns
    -------
    List[str]
        List of saved file paths (empty if nothing saved).
    """
    saved_paths: List[str] = []
    plots = [
        ("Distance from center at touchdown (cm)", [d * 100.0 for d in DISTANCES], "distance_hist.png", "Distance (cm)"),
        ("Touchdown time (s)", TIMES, "time_hist.png", "Time (s)"),
        ("Vertical speed at touchdown (+down, m/s)", VEL_Z, "velz_hist.png", "Vertical speed (m/s)"),
    ]
    for title, arr, fname, xlabel in plots:
        if not arr:
            continue
        plt.figure()
        plt.hist(arr, bins=20)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Count")
        plt.grid(True)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            out = os.path.join(save_dir, fname)
            plt.savefig(out, dpi=150, bbox_inches="tight")
            saved_paths.append(os.path.abspath(out))
        if show:
            plt.show()
        else:
            plt.close()
    return saved_paths

