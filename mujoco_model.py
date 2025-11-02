# mujoco_model.py
"""
Small utility for landing a MuJoCo model from an XML file path.

This module exposes a single function, 'load_model', which:
- Expends user/home shortcuts and resolves the path.
- Validates that the file exists.
- Compiles the XML into a 'mujoco.MjModel', raising clear exceptions on failure.
"""
from __future__ import annotations
from pathlib import Path
import mujoco

def load_model(xml_path: str | Path) -> mujoco.MjModel:
    """
    Load and compile a MuJoCo model from an XML file.
    :param xml_path: str | pathlib.Path
        Path to the MuJoCo XML file. User home shortcuts are supported.
    :return:
    mujoco.MjModel
        The compiled MuJoCo model.

    :raises
    FileNotFoundError
        If the provided path does not exist.
    RuntimeError
        If MuJoCo fails to parse or compile the XML.
    """
    path = Path(xml_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Scene XML not found: {path}")
    try:
        return mujoco.MjModel.from_xml_path(str(path))
    except Exception as e:
        raise RuntimeError(f"Failed to parse/compile XML: {path}\n{e}") from e


