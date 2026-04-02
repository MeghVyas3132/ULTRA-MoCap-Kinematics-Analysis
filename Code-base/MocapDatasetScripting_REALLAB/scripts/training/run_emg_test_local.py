#!/usr/bin/env python3
"""
Quick local test of EMG improvements
Tests on 1 subject with reduced epochs to verify everything works
"""

import os
import sys

# Setup paths
repo_dir = '/Users/meghvyas/Desktop/research-paper'
os.chdir(repo_dir)
sys.path.insert(0, f'{repo_dir}/Code-base/MocapDatasetScripting_REALLAB/scripts/training')

# Configuration
os.environ['DATA_H5_PATH'] = f'{repo_dir}/Dataset/ULTra-MoCap-processed/All_subjects_data.h5'
os.environ['DATASET_ROOT'] = f'{repo_dir}/Code-base/MocapDatasetScripting_REALLAB/datasets'
os.environ['RESULTS_FOLDER'] = f'{repo_dir}/Code-base/MocapDatasetScripting_REALLAB/results/EMG_test_local'
os.environ['RESULT_TAG'] = 'local_test'

# Quick test settings
os.environ['MAX_FOLDS'] = '1'  # Just 1 subject
os.environ['MODALITIES'] = 'emg'
os.environ['EMG_MODEL_VARIANT'] = 'conformer'
os.environ['EMG_EPOCHS'] = '5'  # Just 5 epochs for quick test
os.environ['EMG_PATIENCE'] = '3'
os.environ['EMG_LR'] = '5e-4'
os.environ['EMG_BATCH_SIZE'] = '64'
os.environ['USE_AUGMENTATION'] = '1'
os.environ['USE_TTA'] = '1'
os.environ['USE_LABEL_SMOOTHING'] = '1'
os.environ['LABEL_SMOOTHING'] = '0.1'

# Performance settings
os.environ['DATALOADER_WORKERS'] = '2'
os.environ['PREFETCH_FACTOR'] = '2'

print("=" * 70)
print("EMG IMPROVEMENTS - LOCAL QUICK TEST")
print("=" * 70)
print(f"\nConfiguration:")
print(f"  Device: MPS (Apple Silicon GPU)")
print(f"  Model: Conformer")
print(f"  Folds: 1 (quick test)")
print(f"  Epochs: 5 (quick test)")
print(f"  Augmentation: Enabled")
print(f"  TTA: Enabled")
print(f"  Label Smoothing: 0.1")
print(f"\nThis will take ~15-30 minutes to verify everything works.")
print("=" * 70)
print()

# Import and run the optimized LOSO script
try:
    import emg_optimized_loso
    print("\n" + "=" * 70)
    print("✅ LOCAL TEST COMPLETED SUCCESSFULLY!")
    print("=" * 70)
except Exception as e:
    print(f"\n❌ Error during training: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
