"""Backward-compatible launcher for the reorganized subject visualization script."""

from pathlib import Path
import runpy

TARGET = Path(__file__).resolve().parent / "scripts" / "visualization" / "visualize_subject_distribution.py"
runpy.run_path(str(TARGET), run_name="__main__")
