# visualization.py
# -*- coding: utf-8 -*-
"""
Full 3D visualization using GLFW + MuJoCo mjv/mjr (no mujoco.viewer),
runs directly from e.g. PyCharm.

Highlights
----------
- Optional follow-camera mode with a keyboard toggle (F).
- Overlay drawn as a **single consolidated line** (including `note`) to avoid overlap.
- Scene renders to the framebuffer viewport (HiDPI/Retina-aware).
- Text overlays target the window-size viewport (to avoid density/duplication issues).
"""
from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple
import math
import time

import glfw
import mujoco


# --------- Global state ---------
_window: Optional[glfw._GLFWwindow] = None
_win_w, _win_h = 1280, 900  # logical window size
_fb_w, _fb_h = 1280, 900    # framebuffer size in pixels

_viewport_scene   = mujoco.MjrRect(0, 0, _fb_w, _fb_h)   # for scene (FB)
_viewport_overlay = mujoco.MjrRect(0, 0, _win_w, _win_h) # for text (window)

_model: Optional[mujoco.MjModel] = None
_data: Optional[mujoco.MjData] = None

_cam = mujoco.MjvCamera()
_opt = mujoco.MjvOption()
_scene: Optional[mujoco.MjvScene] = None
_context: Optional[mujoco.MjrContext] = None

_status_text: Optional[str] = None
_overlay: Dict[str, Any] = {}

_DRONE_BODY_NAME: Optional[str] = None
_PLATFORM_TOP_Z: float = 0.0
_WINDOW_TITLE: str = "Drone Landing (MuJoCo)"

# NEW: follow flags
_follow_enabled: bool = True  # can be changed via init/keyboard/API

# Enum shortcuts for Python API (mjr_overlay expects SupportsInt)
FS_150 = int(mujoco.mjtFontScale.mjFONTSCALE_150)
FS_100 = int(mujoco.mjtFontScale.mjFONTSCALE_100)
GP_TL  = int(mujoco.mjtGridPos.mjGRID_TOPLEFT)
GP_BR  = int(mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT)   # consolidated row drawn here (bottom-right)

# --------- Helpers ---------
def _fmt_float(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 100 else f"{v:.1f}"
    return str(v)

def _compose_single_line() -> str:
    """
    Build a single, unified telemetry row (including `note` at the end),
    and shorten it if it becomes too long. We deliberately show only one row
    to avoid overlap—even on HiDPI screens.
    """
    if not _overlay:
        return ""

    # Display ex/ey too if supplied by the runner; no new computation here.
    fields_order = [
        "mode", "steps", "sim_time",
        "dxy", "ex", "ey",        # ← horizontal values: norm and components
        "dz", "vz",
        "clr", "clr_tol"
    ]
    parts: List[str] = []
    for k in fields_order:
        if k in _overlay:
            parts.append(f"{k}={_fmt_float(_overlay[k])}")

    # `note` at the end (if present)
    if "note" in _overlay and _overlay["note"]:
        parts.append(f"note={str(_overlay['note'])}")

    # Extra fields not listed above
    for k, v in _overlay.items():
        if k in fields_order or k == "note":
            continue
        parts.append(f"{k}={_fmt_float(v)}")

    line = ", ".join(parts)

    # Shorten if too long (rough estimate: ~90 chars for FS_100)
    MAX_CHARS = 90
    if len(line) > MAX_CHARS:
        line = line[:MAX_CHARS - 1] + "…"

    return line

# --------- Window events ---------
def _update_viewports(win) -> None:
    """Scene uses FB size; text uses window size. Solves overlap/duplication on HiDPI."""
    global _fb_w, _fb_h, _win_w, _win_h, _viewport_scene, _viewport_overlay
    _win_w, _win_h = glfw.get_window_size(win)
    _fb_w, _fb_h   = glfw.get_framebuffer_size(win)
    _viewport_scene   = mujoco.MjrRect(0, 0, int(_fb_w), int(_fb_h))
    _viewport_overlay = mujoco.MjrRect(0, 0, int(_win_w), int(_win_h))

def _on_resize(win, w, h):
    _update_viewports(win)

def _on_key(win, key, scancode, action, mods):
    global _follow_enabled
    if action not in (glfw.PRESS, glfw.REPEAT):
        return
    step_move = 0.05
    step_angle = 3.0
    if key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(win, True)
        return
    # Orbit angles
    if key == glfw.KEY_LEFT:
        _cam.azimuth -= step_angle
    elif key == glfw.KEY_RIGHT:
        _cam.azimuth += step_angle
    elif key == glfw.KEY_UP:
        _cam.elevation = min(_cam.elevation + step_angle, 89.0)
    elif key == glfw.KEY_DOWN:
        _cam.elevation = max(_cam.elevation - step_angle, -89.0)
    # Zoom
    elif key == glfw.KEY_MINUS:
        _cam.distance *= 1.05
    elif key == glfw.KEY_EQUAL:
        _cam.distance /= 1.05
    # Pan (WASD/QE)
    elif key in (glfw.KEY_W, glfw.KEY_S, glfw.KEY_A, glfw.KEY_D, glfw.KEY_Q, glfw.KEY_E):
        az = math.radians(_cam.azimuth)
        fwd = (math.cos(az), math.sin(az))
        right = (math.cos(az + math.pi/2.0), math.sin(az + math.pi/2.0))
        dx = dy = dz = 0.0
        if key == glfw.KEY_W: dx += fwd[0]*step_move; dy += fwd[1]*step_move
        if key == glfw.KEY_S: dx -= fwd[0]*step_move; dy -= fwd[1]*step_move
        if key == glfw.KEY_D: dx += right[0]*step_move; dy += right[1]*step_move
        if key == glfw.KEY_A: dx -= right[0]*step_move; dy -= right[1]*step_move
        if key == glfw.KEY_E: dz += step_move
        if key == glfw.KEY_Q: dz -= step_move
        _cam.lookat[0] += dx
        _cam.lookat[1] += dy
        _cam.lookat[2] += dz
    # NEW: toggle follow mode (F)
    elif key == glfw.KEY_F and action == glfw.PRESS:
        _follow_enabled = not _follow_enabled
        # brief status hint (optional)
        set_status_text(f"Follow: {'ON' if _follow_enabled else 'OFF'}")

# --------- Public API ---------
def init(model: mujoco.MjModel,
         data: mujoco.MjData,
         drone_body: str = "cf2",
         platform_top_z: float = 0.0,
         window_title: str = "Drone Landing (MuJoCo)",
         *,
         follow: bool = True) -> None:
    """
    Create a GLFW window and wire up mjv/mjr. This does not use `mujoco.viewer`
    (so it runs directly from IDEs like PyCharm).

    Parameters
    ----------
    model, data : MuJoCo handles
    drone_body : str
        Name of the drone body (used for follow-camera targeting).
    platform_top_z : float
        Reserved for future HUD elements (not used internally here).
    window_title : str
        Title for the created window.
    follow : bool
        If True (default), a soft follow camera tracks the drone. If False,
        you get a free camera; use arrows/WASD/QE, and toggle follow anytime with F.
    """
    global _window, _scene, _context, _model, _data
    global _DRONE_BODY_NAME, _PLATFORM_TOP_Z, _WINDOW_TITLE, _follow_enabled

    _model, _data = model, data

    if not glfw.init():
        raise RuntimeError("GLFW init failed (if missing: pip install glfw).")

    glfw.window_hint(glfw.SAMPLES, 4)
    glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
    _window = glfw.create_window(_win_w, _win_h, window_title, None, None)
    if not _window:
        glfw.terminate()
        raise RuntimeError("GLFW create_window failed")

    glfw.make_context_current(_window)
    glfw.swap_interval(1)
    glfw.set_window_size_callback(_window, _on_resize)
    glfw.set_key_callback(_window, _on_key)

    _update_viewports(_window)

    mujoco.mjv_defaultCamera(_cam)
    mujoco.mjv_defaultOption(_opt)

    # Disable distracting debug visuals
    _opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
    _opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 0
    _opt.frame = mujoco.mjtFrame.mjFRAME_NONE

    _DRONE_BODY_NAME = drone_body
    _PLATFORM_TOP_Z = float(platform_top_z)
    _WINDOW_TITLE = str(window_title)
    _follow_enabled = bool(follow)

    # Initial camera
    _cam.distance = 1.6
    _cam.elevation = -25.0
    _cam.azimuth = 90.0
    try:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, drone_body)
        _cam.lookat[:] = data.xpos[bid]
        _cam.lookat[2] += 0.05
    except Exception:
        pass

    _scene = mujoco.MjvScene(model, maxgeom=20000)
    _context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

def should_close() -> bool:
    """Return True if the window was requested to close (ESC or close button)."""
    return (_window is None) or glfw.window_should_close(_window)

def set_status_text(text: Optional[str]) -> None:
    """Set a large banner text (top-left). Pass None or '' to clear."""
    global _status_text
    _status_text = text if (text is None or isinstance(text, str)) else str(text)

def set_overlay_data(**kwargs) -> None:
    """Replace the single-line telemetry dictionary with the provided keyword fields."""
    global _overlay
    _overlay = dict(kwargs) if kwargs else {}


# Optional: soft follow-camera tracking of the drone
def _maybe_follow_drone():
    """If follow mode is on, softly move the camera lookat toward the drone body."""
    if not _follow_enabled:
        return
    if _model is None or _data is None or _DRONE_BODY_NAME is None:
        return
    try:
        bid = mujoco.mj_name2id(_model, mujoco.mjtObj.mjOBJ_BODY, _DRONE_BODY_NAME)
        px, py, pz = _data.xpos[bid]
        alpha = 0.15
        _cam.lookat[0] = (1-alpha)*_cam.lookat[0] + alpha*float(px)
        _cam.lookat[1] = (1-alpha)*_cam.lookat[1] + alpha*float(py)
        _cam.lookat[2] = (1-alpha)*_cam.lookat[2] + alpha*(float(pz) + 0.05)
    except Exception:
        pass

def _render_scene():
    """Update mjv scene graph and render to the framebuffer viewport."""
    assert _scene is not None and _context is not None and _model is not None and _data is not None
    mujoco.mjv_updateScene(_model, _data, _opt, None, _cam, mujoco.mjtCatBit.mjCAT_ALL, _scene)
    mujoco.mjr_render(_viewport_scene, _scene, _context)

def _render_hud():
    """Draw overlay: banner at top-left and single-line telemetry at bottom-right."""
    if _context is None:
        return
    # Top-left banner (e.g., "LANDED ✅")
    if _status_text:
        mujoco.mjr_overlay(FS_150, GP_TL, _viewport_overlay, str(_status_text), "", _context)

    # Single consolidated telemetry row at bottom-right
    line = _compose_single_line()
    if line:
        mujoco.mjr_overlay(FS_100, GP_BR, _viewport_overlay, line, "", _context)

def render_once() -> None:
    """
    Render one frame: scene + HUD. Call from the simulation loop after physics updates.
    """
    if _window is None or _scene is None or _context is None:
        time.sleep(1.0 / 120.0)
        return

    # Update viewports (HiDPI/resize/scale)
    _update_viewports(_window)

    # Soft follow camera (only if follow is active)
    _maybe_follow_drone()

    # Background color (by FB size)
    mujoco.mjr_rectangle(_viewport_scene, 0.02, 0.02, 0.025, 1.0)

    # Draw
    _render_scene()
    _render_hud()

    glfw.swap_buffers(_window)
    glfw.poll_events()

def cleanup() -> None:
    """Destroy GL/MuJoCo resources and the GLFW window; terminate GLFW."""
    global _window, _scene, _context, _model, _data
    try:
        if _context is not None:
            _context.free()
    except Exception:
        pass
    try:
        if _scene is not None:
            _scene.free()
    except Exception:
        pass
    _context = None
    _scene = None
    _model = None
    _data = None

    if _window is not None:
        try:
            glfw.destroy_window(_window)
        except Exception:
            pass
        _window = None
    try:
        glfw.terminate()
    except Exception:
        pass

def enable_fog(model, color=(0.7, 0.7, 0.75, 1.0), start=None, end=None):
    """
    Enable fog in rendering.

    Parameters
    ----------
    model : mujoco.MjModel
        Visual parameters will be written into this model (vis/map).
    color : tuple[float, float, float, float]
        RGBA fog color the scene fades toward.
    start, end : float | None
        Distances (meters) where fog starts/ends. If None, use scene extent.

    """
    global _scene

    # Defaults based on scene scale
    extent = float(model.stat.extent)
    if start is None:
        start = 0.7 * extent
    if end is None:
        end = 2.5 * extent

    # Set fog color & ranges (equiv. to <visual><rgba fog=".."/><map fogstart=".." fogend=".."/>)
    c = (list(color) + [1.0, 1.0, 1.0, 1.0])[:4]  # ensure 4-length RGBA
    for i in range(4):
        model.vis.rgba.fog[i] = float(c[i])
    model.vis.map.fogstart = float(start)
    model.vis.map.fogend   = float(end)

    # Scene render flag for fog (GLFW + mjr/mjv)
    if _scene is not None:
        _scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1
