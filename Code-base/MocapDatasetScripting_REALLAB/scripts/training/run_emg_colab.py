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

# Import and run the training script
try:
    # Use improved script if EMG_MODEL_VARIANT is conformer or dual_branch
    use_improved = EMG_MODEL_VARIANT.lower() in ['conformer', 'dual_branch']
    
    if use_improved:
        script_name = 'conv1d_bigru_loso_improved.py'
        print(f"✓ Using IMPROVED training script for {EMG_MODEL_VARIANT} model\n")
    else:
        script_name = 'conv1d_bigru_loso.py'
        print(f"✓ Using standard training script for {EMG_MODEL_VARIANT} model\n")
    
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    
    if not os.path.exists(script_path):
        print(f"❌ ERROR: Training script not found: {script_path}")
        # Fall back to standard script
        if use_improved:
            print("⚠️ Falling back to standard script...")
            script_path = os.path.join(os.path.dirname(__file__), 'conv1d_bigru_loso.py')
            use_improved = False
    
    # For improved script, we need to run it as a subprocess to properly override model creation
    # The exec() approach doesn't work because the improved script uses "from conv1d_bigru_loso import *"
    # which causes namespace conflicts
    if use_improved:
        import subprocess
        result = subprocess.run([sys.executable, script_path], cwd=os.path.dirname(script_path))
        if result.returncode != 0:
            print(f"\n⚠️ Improved script failed with code {result.returncode}")
            print("This is expected - improved script needs to be run directly")
            sys.exit(result.returncode)
    else:
        # Standard script can be exec'd
        old_argv = sys.argv
        sys.argv = [script_name]
        
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
