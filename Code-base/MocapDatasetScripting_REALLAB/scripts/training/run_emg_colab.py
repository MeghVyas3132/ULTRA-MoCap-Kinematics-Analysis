#!/usr/bin/env python3
"""
Simple EMG training wrapper for Google Colab
This script runs EMG-only training with improvements from emg_improvements.py
"""

import os
import sys

# Get configuration from environment
H5_PATH = os.getenv('DATA_H5_PATH', 'Dataset/ULTra-MoCap-processed/All_subjects_data.h5')
MAX_FOLDS = int(os.getenv('MAX_FOLDS', '0'))
EMG_EPOCHS = int(os.getenv('EMG_EPOCHS', '50'))
EMG_MODEL_VARIANT = os.getenv('EMG_MODEL_VARIANT', 'conformer')

print("="*70)
print("EMG TRAINING - COLAB WRAPPER")
print("="*70)
print(f"Dataset: {H5_PATH}")
print(f"Folds: {'All 13' if MAX_FOLDS == 0 else MAX_FOLDS}")
print(f"Epochs: {EMG_EPOCHS}")
print(f"Model: {EMG_MODEL_VARIANT}")
print("="*70)

# Check if dataset exists
if not os.path.exists(H5_PATH):
    print(f"\n❌ ERROR: Dataset not found at {H5_PATH}")
    print("\nPlease ensure:")
    print("1. Dataset is uploaded to Google Drive")
    print("2. Drive is mounted (run Cell 2)")
    print("3. DATA_H5_PATH environment variable is correct")
    sys.exit(1)

# Check if emg_improvements module is available
try:
    sys.path.insert(0, os.path.dirname(__file__))
    import emg_improvements
    print("\n✓ EMG improvements module loaded")
except ImportError as e:
    print(f"\n⚠️ Warning: Could not import emg_improvements: {e}")
    print("Proceeding with standard training...")

# Now run the main training script
print("\n" + "="*70)
print("LAUNCHING TRAINING...")
print("="*70 + "\n")

# Import and run conv1d_bigru_loso
try:
    # The conv1d_bigru_loso.py script is designed to be run as main
    # We'll execute it using exec to run it in this process
    script_path = os.path.join(os.path.dirname(__file__), 'conv1d_bigru_loso.py')
    
    if not os.path.exists(script_path):
        print(f"❌ ERROR: Training script not found: {script_path}")
        sys.exit(1)
    
    # Set sys.argv for the script
    old_argv = sys.argv
    sys.argv = ['conv1d_bigru_loso.py']
    
    try:
        with open(script_path, 'r') as f:
            code = f.read()
        exec(code, {'__name__': '__main__', '__file__': script_path})
    finally:
        sys.argv = old_argv
        
except Exception as e:
    print(f"\n❌ ERROR during training: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✅ TRAINING COMPLETED")
print("="*70)
