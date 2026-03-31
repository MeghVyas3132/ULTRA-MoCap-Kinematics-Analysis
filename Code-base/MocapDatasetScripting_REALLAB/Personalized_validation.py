"""Backward-compatible launcher for the reorganized personalized validation script."""

from pathlib import Path
import runpy

TARGET = Path(__file__).resolve().parent / "scripts" / "training" / "personalized_validation.py"
runpy.run_path(str(TARGET), run_name="__main__")
