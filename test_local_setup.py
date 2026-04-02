#!/usr/bin/env python3
"""Quick test before running full training"""

import sys
from pathlib import Path

print("🔍 Testing local setup...\n")

# 1. Check Python
print(f"✓ Python: {sys.version.split()[0]}")

# 2. Check PyTorch
try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ MPS available: {torch.backends.mps.is_available()}")
except ImportError:
    print("❌ PyTorch not installed!")
    sys.exit(1)

# 3. Check dataset
h5_path = Path("Dataset/ULTra-MoCap-processed/All_subjects_data.h5")
if h5_path.exists():
    size_gb = h5_path.stat().st_size / (1024**3)
    print(f"✓ Dataset found: {size_gb:.2f} GB")
else:
    print("❌ Dataset not found!")
    sys.exit(1)

# 4. Check training scripts
scripts_dir = Path("Code-base/MocapDatasetScripting_REALLAB/scripts/training")
run_emg = scripts_dir / "run_emg_colab.py"
emg_imp = scripts_dir / "emg_improvements.py"

print(f"✓ Training script: {run_emg.exists()}")
print(f"✓ Improvements: {emg_imp.exists()}")

# 5. Test import
sys.path.insert(0, str(scripts_dir))
try:
    import emg_improvements
    print(f"✓ emg_improvements module loaded")
except ImportError as e:
    print(f"⚠️ Could not import emg_improvements: {e}")

# 6. Check dependencies
deps = ['numpy', 'pandas', 'scipy', 'sklearn', 'h5py', 'pywt']
missing = []
for dep in deps:
    try:
        __import__(dep)
        print(f"✓ {dep}")
    except ImportError:
        print(f"❌ {dep} missing")
        missing.append(dep)

if missing:
    print(f"\n⚠️ Install missing: pip install {' '.join(missing)}")
else:
    print("\n✅ All checks passed! Ready to train.")

