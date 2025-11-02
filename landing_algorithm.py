# landing_algorithm.py
"""
Landing Algorithm:
velocity-command landing controller for a quadrotot
to align above a (possibly moving) platform and descend safely.

Design:
The controller runs in three conceptual behaviors (implicitly blended):
1) CHASE - close horizontal error (relative XY to platform center).
2) ALIGN - hold above platform (small XY error & small relative XY velocity).
3) DESCEND - reduce altitude while maintaining alignment and safe vertical rates.

Units & Frames
- Positions: meters (m) in world frame
- Velocities: meters/seconds (m/s) in world frame
- Command: desired world-frame linear velocities [vx, vy, vz]
"""
import math

# ====== Tuning (XY) ======

# ----- XY pursuit (CHASE) -----
kp_xy = 0.9 # P gain for horizontal position error (m->m/s)
kd_xy = 0.9 # D gain for relative horizontal velocity (m/s->m/s)

# ----- XY hold (ALIGN) -----
kp_hold_xy = 0.25 # gentler P gain when close to center
kd_hold_xy = 1.2  # stronger damping to match platform motion

# ----- XY speed limits & blending -----
VXY_MAX_FAR  = 10.0       # cap when far from the platform
VXY_MAX_NEAR = 0.9        # tighter cap when near (prevents overshoot)
ALT_SPEED_BLEND = 0.6     # [m] below this altitude, restrict XY speed toward VXY_MAX_NEAR

# ----- Alignment detection -----
ALIGN_RADIUS = 0.20       # [m] within this horizontal radius considered "centered enough"
ALIGN_VREL   = 0.10       # [m/s] relative XY speed threshold for "steady enough"

# ----- Vertical descent -----
DESCENT_START_RADIUS = 2.0    # [m] below this, begin permitting descent
DESCENT_ALLOW_RADIUS = 0.2    # [m] allow full descent only below this radius
DESCENT_VREL_GATE    = 0.20   # [m/s] must match platform velocity before allowing significant descent
STOP_DESCENT_RADIUS  = 0.40   # [m] if we moved farther than this-stoop descending (hysteresis to prevent chatter)

# ====== Tuning (Z) ======
kp_z   = 1.0                 # P gain for altitude (m -> m/s downward when positive)
VZ_MAX = 0.6                 # [m/s] maximum allowed descent rate when high
VZ_NEAR = 0.20               # [m/s] descent cap when near ground
ALT_NEAR = 0.20              # [m] below this altitude, restrict vertical speed
FLARE_ALT = 0.10             # [m] start flare (extra slow) close to touchdown
VZ_FLARE  = 0.12             # [m/s] descent cap during flare

def clamp(v, lo, hi):
    """Clamp val into [lo,hi]."""
    return max(lo, min(hi, v))

def limit_vec2(x, y, vmax):
    """
    Limit a 2D vector's magnitude to 'vmax' (preserving direction).
    float: param x, y: Vector components in m/s.
    float: param vmax: Maximum allowed magnitude (m/s).
    (float, float): return: Possibly scaled (x, y) so that sqrt (x^2 + y^2) <= vmax.
    """
    n = math.hypot(x, y)
    if n <= vmax or n == 0.0:
        return x, y
    s = vmax / n
    return x*s, y*s

def _smooth_blend(x0, x1, x):
    """Returns 0..1 smoothly as x moves between x0 (0) and x1 (1)."""
    if x <= x0: return 0.0
    if x >= x1: return 1.0
    u = (x - x0) / (x1 - x0)
    return u*u*(3 - 2*u)  # smoothstep

def _descent_gate(dist, vrel_xy, altitude):
    """
    Determine how "open" to permit descent (0..1) based on hovering above center and velocity matching.

    Logic:
    -Position gate: gradual transition from 0 (far) to 1 (near) between DESCENT_START_RADIUS and DESCENT_ALLOW_RADIUS.
    -Relative-speed gate: require low relative XY speed before significant descent.
    -Near the ground (FLARE_ALT): apply a small softening to avoid sharp dives.

    Parameters
    float: dist: Horizontal distance from platform center [m].
    float: vrel_xy: Relative horizontal speed between drone and platform [m/s].
    float: altitude: "Altitude to land": >0 when the drone bottom is above the platform top [m].

    :returns
    float: Gate in [0..1] that scales the vertical command (larger -> more "open" to descend).
    """
    # Position gate: between DESCENT_START_RADIUS -> DESCENT_ALLOW_RADIUS (inverted)
    open_by_pos = _smooth_blend(DESCENT_START_RADIUS, DESCENT_ALLOW_RADIUS, DESCENT_START_RADIUS - dist)  # inverted
    # Relative speed gate: must have low v_rel
    open_by_vrel = 1.0 - clamp(vrel_xy / DESCENT_VREL_GATE, 0.0, 1.0)
    gate = clamp(open_by_pos * open_by_vrel, 0.0, 1.0)

    # flare near ground
    if altitude <= FLARE_ALT:
        gate *= 0.6  # small softening
    return gate

def make_decision(
    dist, dist_x, dist_y, altitude,
    rel_vel_x, rel_vel_y,
    vplat_x, vplat_y,   # Platform velocity feed-forward
):
    """
    Returns world-frame velocity commands: (v_cmd_x, v_cmd_y, v_cmd_z)

    Key idea:
    v_cmd_xy_world = v_platform_xy + v_rel_correction(dist, rel_vel)
    This "rides" with the platform while closing the relative error.

    Parameters
    float: dist: Total horizontal distance from platform center [m].
    float: dist_x, dist_y: Relative position error along X,Y (drone - platform) [m].
    float: altitude: Altitude-to-land: >0 when drone is above platform top [m].
    float: rel_vel_x, rel_vel_y: Relative horizontal velocity (drone - platform) [m/s].
    float: vplat_x, vplat_y: Platform world velocity [m/s] (feed-forward).

    :returns
    (float, float, float): (v_cmd_x, v_cmd_y, v_cmd_z) - world-frame velocity commands [m/s].
    """

    # Horizontal limit as a function of altitude
    blend_alt = clamp(altitude / ALT_SPEED_BLEND, 0.0, 1.0)
    vxy_allowed = VXY_MAX_NEAR + (VXY_MAX_FAR - VXY_MAX_NEAR) * blend_alt

    # Alignment checks
    aligned_pos = (dist <= ALIGN_RADIUS)
    aligned_vel = (abs(rel_vel_x) <= ALIGN_VREL and abs(rel_vel_y) <= ALIGN_VREL)
    aligned = aligned_pos and aligned_vel

    # Relative correction (not world)
    if aligned:
        # Gentle hold over center during descent
        u_rel_x = -(kd_hold_xy * rel_vel_x + kp_hold_xy * dist_x)
        u_rel_y = -(kd_hold_xy * rel_vel_y + kp_hold_xy * dist_y)
    else:
        # Regular PD for relative horizontal approach
        u_rel_x = -(kp_xy * dist_x + kd_xy * rel_vel_x)
        u_rel_y = -(kp_xy * dist_y + kd_xy * rel_vel_y)

    # feed-forward: add platform velocity to "ride" with it
    v_cmd_x = vplat_x + u_rel_x
    v_cmd_y = vplat_y + u_rel_y
    v_cmd_x, v_cmd_y = limit_vec2(v_cmd_x, v_cmd_y, vxy_allowed)

    # Descent gates (Z)
    vrel_xy = math.hypot(rel_vel_x, rel_vel_y)
    gate = _descent_gate(dist, vrel_xy, altitude)

    # Stop descending if we drifted away again
    if dist > STOP_DESCENT_RADIUS:
        gate *= 0.0

    # Allowed descent rate vs altitude
    vz_cap = VZ_NEAR if altitude <= ALT_NEAR else VZ_MAX
    if altitude <= FLARE_ALT:
        vz_cap = min(vz_cap, VZ_FLARE)

    # Vertical velocity - negative is downward
    v_z_cmd = -clamp(kp_z * max(altitude, 0.0), 0.0, vz_cap) * gate

    # If fully aligned - allow full descent within safety caps
    if aligned:
        v_z_cmd = -clamp(kp_z * max(altitude, 0.0), 0.0, vz_cap)

    return float(v_cmd_x), float(v_cmd_y), float(v_z_cmd)
