#!/usr/bin/env python3
"""
Local EMG Training with Live Logs
Run with: python run_local_training.py

This runner launches the main, maintained LOSO script directly.
"""

import os
import sys
import subprocess
from pathlib import Path

# Setup paths
REPO_DIR = Path(__file__).parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / 'Code-base' / 'MocapDatasetScripting_REALLAB'))

# Configuration
H5_PATH = str(REPO_DIR / 'Dataset' / 'ULTra-MoCap-processed' / 'All_subjects_data.h5')
RESULTS_DIR = str(REPO_DIR / 'Code-base' / 'MocapDatasetScripting_REALLAB' / 'results')

# Training settings
# Supported variants: convbigru, lstm_msa, conformer, dual_branch
EMG_MODEL_VARIANT = os.getenv('EMG_MODEL_VARIANT', 'conformer')
EMG_EPOCHS = int(os.getenv('EMG_EPOCHS', '50'))
MAX_FOLDS = int(os.getenv('MAX_FOLDS', '0'))  # 0 = all 13, or 1-3 for testing

print("="*70)
print("🚀 LOCAL EMG TRAINING - LIVE LOGS")
print("="*70)
print(f"Dataset: {H5_PATH}")
print(f"Model: {EMG_MODEL_VARIANT}")
print(f"Folds: {'All 13' if MAX_FOLDS == 0 else MAX_FOLDS}")
print(f"Epochs: {EMG_EPOCHS}")
print(f"Device: MPS (Apple Silicon GPU)")
print("="*70 + "\n")

# Check dataset exists
if not Path(H5_PATH).exists():
    print(f"❌ ERROR: Dataset not found at {H5_PATH}")
    sys.exit(1)

# Find training script
training_script = REPO_DIR / 'Code-base' / 'MocapDatasetScripting_REALLAB' / 'scripts' / 'training' / 'conv1d_bigru_loso.py'
if not training_script.exists():
    print(f"❌ ERROR: Training script not found: {training_script}")
    sys.exit(1)

print(f"✓ Using script: {training_script.name}\n")

# Environment variables
env = os.environ.copy()
env.update({
    'DATA_H5_PATH': H5_PATH,
    'MAX_FOLDS': str(MAX_FOLDS),
    'EMG_EPOCHS': str(EMG_EPOCHS),
    'EMG_MODEL_VARIANT': EMG_MODEL_VARIANT,
    'MODALITIES': 'emg',
    'DATALOADER_WORKERS': env.get('DATALOADER_WORKERS', '0'),
    'PYTHONUNBUFFERED': '1',  # Immediate output
})

# Run training with LIVE output (no capture)
print("Starting training...\n")
try:
    subprocess.run(
        [sys.executable, '-u', str(training_script)],
        cwd=str(REPO_DIR),
        env=env,
        check=True
    )
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETED!")
    print("="*70)
except subprocess.CalledProcessError as e:
    print(f"\n❌ Training failed with error code {e.returncode}")
    sys.exit(1)
except KeyboardInterrupt:
    print("\n⚠️ Training interrupted by user")
    sys.exit(0)
