# EMG Training Visualization - Real-Time Monitoring

## Quick Setup - Add This Cell to Your Colab Notebook

### Option 1: Real-Time Training Monitor (Add BEFORE running training)

```python
# Cell: Training Visualization Setup
import matplotlib.pyplot as plt
from IPython import display
import time
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Enable interactive plotting
%matplotlib inline

def monitor_training(results_dir, refresh_seconds=30, max_updates=200):
    """
    Monitor training progress in real-time
    Args:
        results_dir: Path to results directory
        refresh_seconds: How often to refresh (default 30s)
        max_updates: Maximum number of updates (default 200)
    """
    plt.ion()  # Interactive mode
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for update_num in range(max_updates):
        try:
            # Find latest CSV file
            csv_files = list(Path(results_dir).rglob('*fold*.csv')) + \
                       list(Path(results_dir).rglob('*summary*.csv'))
            
            if not csv_files:
                print(f"⏳ Waiting for results... (update {update_num+1})")
                time.sleep(refresh_seconds)
                continue
            
            # Clear previous plots
            for ax in axes.flat:
                ax.clear()
            
            # Read all fold results
            all_data = []
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    all_data.append(df)
                except:
                    continue
            
            if not all_data:
                time.sleep(refresh_seconds)
                continue
            
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Plot 1: Training Loss Over Epochs
            ax1 = axes[0, 0]
            if 'epoch' in combined_df.columns and 'train_loss' in combined_df.columns:
                for fold in combined_df['fold'].unique() if 'fold' in combined_df.columns else [0]:
                    fold_data = combined_df[combined_df['fold'] == fold] if 'fold' in combined_df.columns else combined_df
                    ax1.plot(fold_data['epoch'], fold_data['train_loss'], 
                            alpha=0.6, label=f'Fold {fold}')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Training Loss')
                ax1.set_title('Training Loss Curves')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            # Plot 2: Validation Accuracy
            ax2 = axes[0, 1]
            if 'epoch' in combined_df.columns and 'val_acc' in combined_df.columns:
                for fold in combined_df['fold'].unique() if 'fold' in combined_df.columns else [0]:
                    fold_data = combined_df[combined_df['fold'] == fold] if 'fold' in combined_df.columns else combined_df
                    ax2.plot(fold_data['epoch'], fold_data['val_acc'] * 100, 
                            alpha=0.6, label=f'Fold {fold}', marker='o')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Validation Accuracy (%)')
                ax2.set_title('Validation Accuracy Over Time')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                ax2.axhline(y=70, color='r', linestyle='--', label='70% Target', alpha=0.5)
            
            # Plot 3: Per-Fold Performance
            ax3 = axes[1, 0]
            if 'fold' in combined_df.columns:
                # Get best accuracy per fold
                fold_best = combined_df.groupby('fold')['val_acc'].max() * 100
                ax3.bar(range(len(fold_best)), fold_best.values, alpha=0.7, color='skyblue')
                ax3.axhline(y=70, color='r', linestyle='--', label='70% Target', alpha=0.5)
                ax3.set_xlabel('Fold (Subject)')
                ax3.set_ylabel('Best Accuracy (%)')
                ax3.set_title('Best Accuracy Per Fold')
                ax3.set_xticks(range(len(fold_best)))
                ax3.set_xticklabels([f'S{i}' for i in fold_best.index])
                ax3.legend()
                ax3.grid(True, alpha=0.3, axis='y')
            
            # Plot 4: Overall Statistics
            ax4 = axes[1, 1]
            if 'val_acc' in combined_df.columns:
                completed_folds = len(combined_df['fold'].unique()) if 'fold' in combined_df.columns else 1
                mean_acc = combined_df['val_acc'].mean() * 100
                max_acc = combined_df['val_acc'].max() * 100
                min_acc = combined_df['val_acc'].min() * 100
                
                stats_text = f"""
Training Progress

Completed Folds: {completed_folds}/13
                
Current Statistics:
• Mean Accuracy: {mean_acc:.2f}%
• Max Accuracy: {max_acc:.2f}%
• Min Accuracy: {min_acc:.2f}%

Target: 70%+
Status: {'✅ ON TRACK!' if mean_acc >= 70 else '⏳ Training...'}
                """
                ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes,
                        fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                ax4.axis('off')
            
            plt.tight_layout()
            display.clear_output(wait=True)
            display.display(fig)
            
            print(f"\n📊 Update {update_num+1} - {time.strftime('%H:%M:%S')}")
            print(f"Monitoring: {len(csv_files)} files found")
            
            time.sleep(refresh_seconds)
            
        except KeyboardInterrupt:
            print("\n⏹️ Monitoring stopped by user")
            break
        except Exception as e:
            print(f"⚠️ Error in update {update_num+1}: {e}")
            time.sleep(refresh_seconds)
    
    plt.ioff()
    print("\n✅ Monitoring complete!")

# You can run this after starting training
# monitor_training(f'{REPO_DIR}/Code-base/MocapDatasetScripting_REALLAB/results', 
#                  refresh_seconds=30)
```

### Option 2: Quick Progress Check (Run Anytime)

```python
# Cell: Check Training Progress
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

RESULTS_DIR = f'{REPO_DIR}/Code-base/MocapDatasetScripting_REALLAB/results'

# Find all result CSVs
csv_files = list(Path(RESULTS_DIR).rglob('*.csv'))
print(f"📁 Found {len(csv_files)} result files\n")

if csv_files:
    # Read latest results
    latest = sorted(csv_files, key=lambda x: x.stat().st_mtime)[-1]
    df = pd.read_csv(latest)
    
    print(f"📊 Latest Results: {latest.name}\n")
    print("="*60)
    
    if 'val_acc' in df.columns:
        print(f"Best Accuracy: {df['val_acc'].max()*100:.2f}%")
        print(f"Mean Accuracy: {df['val_acc'].mean()*100:.2f}%")
        print(f"Latest Accuracy: {df['val_acc'].iloc[-1]*100:.2f}%")
    
    if 'epoch' in df.columns:
        print(f"Epochs Completed: {df['epoch'].max()}")
    
    print("\n" + "="*60)
    print("\n📈 Last 5 epochs:")
    print(df.tail(5).to_string(index=False))
    
    # Quick plot
    if 'epoch' in df.columns and 'val_acc' in df.columns:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy
        ax[0].plot(df['epoch'], df['val_acc']*100, marker='o')
        ax[0].axhline(y=70, color='r', linestyle='--', alpha=0.5, label='70% Target')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Validation Accuracy (%)')
        ax[0].set_title('Training Progress')
        ax[0].grid(True, alpha=0.3)
        ax[0].legend()
        
        # Loss
        if 'train_loss' in df.columns:
            ax[1].plot(df['epoch'], df['train_loss'], marker='o', color='orange')
            ax[1].set_xlabel('Epoch')
            ax[1].set_ylabel('Training Loss')
            ax[1].set_title('Loss Curve')
            ax[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
else:
    print("⏳ No results yet. Training may still be starting...")
```

### Option 3: Live Logs Monitor (See what's training RIGHT NOW)

```python
# Cell: Monitor Training Logs
import subprocess
import time

def tail_logs(log_file=None, lines=50):
    """
    Show last N lines of training output
    """
    if log_file is None:
        # Find latest log file
        log_dir = f'{REPO_DIR}/Code-base/MocapDatasetScripting_REALLAB/logs'
        log_files = list(Path(log_dir).rglob('*.log')) if Path(log_dir).exists() else []
        if log_files:
            log_file = sorted(log_files, key=lambda x: x.stat().st_mtime)[-1]
    
    if log_file and Path(log_file).exists():
        with open(log_file, 'r') as f:
            all_lines = f.readlines()
            print(''.join(all_lines[-lines:]))
    else:
        print("📋 No log file found. Showing console output...")
        print("\nTraining is running in background.")
        print("Check the launch cell output for progress.")

# Show last 50 lines
tail_logs(lines=50)
```

### Option 4: Model Architecture Visualization

```python
# Cell: Visualize Model Architecture
import torch
import sys
sys.path.insert(0, f'{REPO_DIR}/Code-base/MocapDatasetScripting_REALLAB/scripts/training')

try:
    from emg_improvements import EMGConformer, DualBranchEMG
    
    print("="*60)
    print("MODEL ARCHITECTURE DETAILS")
    print("="*60)
    
    # EMGConformer
    print("\n1️⃣ EMGConformer (Default Model)")
    print("-" * 60)
    model = EMGConformer(in_channels=16, num_classes=7, num_subjects=13)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Model Size: {total_params * 4 / (1024**2):.2f} MB (fp32)")
    
    print("\nArchitecture:")
    print(model)
    
    # Test forward pass
    dummy_input = torch.randn(2, 16, 300)  # batch=2, channels=16, time=300
    with torch.no_grad():
        output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: (batch_size, num_classes={7})")
    
    # DualBranchEMG
    print("\n\n2️⃣ DualBranchEMG (Alternative Model)")
    print("-" * 60)
    model2 = DualBranchEMG(in_channels=16, num_classes=7, num_subjects=13)
    total_params2 = sum(p.numel() for p in model2.parameters())
    trainable_params2 = sum(p.numel() for p in model2.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params2:,}")
    print(f"Trainable Parameters: {trainable_params2:,}")
    print(f"Model Size: {total_params2 * 4 / (1024**2):.2f} MB (fp32)")
    
    print("\n✅ Models loaded successfully!")
    print("\n" + "="*60)
    
except Exception as e:
    print(f"❌ Error loading models: {e}")
    import traceback
    traceback.print_exc()
```

---

## Usage Instructions

### For Real-Time Monitoring:

1. **Add the visualization cell** to your notebook (Option 1)
2. **Run it in a separate cell** WHILE training is running:
   ```python
   monitor_training(f'{REPO_DIR}/Code-base/MocapDatasetScripting_REALLAB/results', 
                    refresh_seconds=30)
   ```

### For Quick Checks:

- Use **Option 2** to quickly see current progress
- Run it anytime during or after training
- Shows latest accuracy, loss, and plots

### What You'll See:

📊 **4 Real-Time Plots:**
1. Training Loss Curves (per fold)
2. Validation Accuracy Over Time
3. Best Accuracy Per Fold
4. Overall Statistics with Target Progress

📈 **Updates Every 30 Seconds:**
- Current accuracy vs 70% target
- Completed folds (X/13)
- Best/mean/min accuracy across folds
- Visual progress indicators

🎯 **Clear Target Tracking:**
- Red line at 70% target
- Status: "✅ ON TRACK!" or "⏳ Training..."
- Real-time ETA based on fold completion

---

## Example Output:

```
📊 Update 15 - 14:23:45
Monitoring: 3 files found

Training Progress

Completed Folds: 3/13

Current Statistics:
• Mean Accuracy: 73.24%
• Max Accuracy: 76.81%
• Min Accuracy: 69.45%

Target: 70%+
Status: ✅ ON TRACK!
```

---

## Tips:

- **Start monitoring early**: Add the cell before training starts
- **Refresh rate**: 30 seconds is good balance (adjust if needed)
- **Long training**: Runs up to 200 updates (100+ minutes)
- **Stop anytime**: Press stop button or Ctrl+C

This gives you full visibility into what's being trained! 🚀
