"""Backward-compatible launcher for the reorganized training script."""

from pathlib import Path
import runpy

TARGET = Path(__file__).resolve().parent / "scripts" / "training" / "conv1d_bigru_loso.py"
runpy.run_path(str(TARGET), run_name="__main__")
