"""Backward-compatible launcher for the reorganized speed visualization script."""

from pathlib import Path
import runpy

TARGET = Path(__file__).resolve().parent / "scripts" / "visualization" / "visualize_speed_distributions.py"
runpy.run_path(str(TARGET), run_name="__main__")
