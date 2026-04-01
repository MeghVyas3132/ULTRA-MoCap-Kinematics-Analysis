# ╔══════════════════════════════════════════════════════════════════╗
# ║                   PASTE THIS INTO NEW COLAB CELL                 ║
# ║               Add AFTER Cell 5, BEFORE running Cell 6            ║
# ╚══════════════════════════════════════════════════════════════════╝

# Cell 5.5: Real-Time Training Visualization
import matplotlib.pyplot as plt
from IPython import display
import time
import os
from pathlib import Path
import pandas as pd
import numpy as np

%matplotlib inline

print("="*70)
print("🎨 TRAINING VISUALIZATION - REAL-TIME MONITOR")
print("="*70)

def monitor_training_live(results_dir, refresh_seconds=30, max_updates=200):
    """
    Monitor EMG training progress in real-time with live plots
    
    Shows:
    - Training loss curves
    - Validation accuracy over time
    - Per-fold performance
    - Progress towards 70% target
    """
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    print(f"\n📊 Monitoring: {results_dir}")
    print(f"🔄 Refresh: every {refresh_seconds} seconds")
    print(f"🎯 Target: 70%+ accuracy\n")
    print("Press STOP button to end monitoring")
    print("="*70 + "\n")
    
    for update_num in range(max_updates):
        try:
            # Find all CSV result files
            csv_files = list(Path(results_dir).rglob('*fold*.csv')) + \
                       list(Path(results_dir).rglob('*results*.csv')) + \
                       list(Path(results_dir).rglob('*summary*.csv'))
            
            if not csv_files:
                print(f"⏳ [{time.strftime('%H:%M:%S')}] Waiting for results... (update {update_num+1}/{max_updates})")
                time.sleep(refresh_seconds)
                continue
            
            # Clear previous plots
            for ax in axes.flat:
                ax.clear()
            
            # Read all available data
            all_data = []
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    if not df.empty:
                        all_data.append(df)
                except:
                    continue
            
            if not all_data:
                time.sleep(refresh_seconds)
                continue
            
            # Combine data
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # ========== PLOT 1: Training Loss ==========
            ax1 = axes[0, 0]
            if 'epoch' in combined_df.columns and 'train_loss' in combined_df.columns:
                folds = combined_df['fold'].unique() if 'fold' in combined_df.columns else [0]
                for fold in sorted(folds)[:13]:  # Max 13 folds
                    fold_data = combined_df[combined_df['fold'] == fold] if 'fold' in combined_df.columns else combined_df
                    if not fold_data.empty:
                        ax1.plot(fold_data['epoch'], fold_data['train_loss'], 
                                alpha=0.7, linewidth=2, label=f'Subject {int(fold)}')
                ax1.set_xlabel('Epoch', fontsize=11)
                ax1.set_ylabel('Training Loss', fontsize=11)
                ax1.set_title('📉 Training Loss Curves', fontsize=13, fontweight='bold')
                ax1.legend(ncol=3, fontsize=8, loc='upper right')
                ax1.grid(True, alpha=0.3)
            else:
                ax1.text(0.5, 0.5, '⏳ Waiting for training data...', 
                        ha='center', va='center', fontsize=14, transform=ax1.transAxes)
                ax1.set_xlim(0, 1)
                ax1.set_ylim(0, 1)
            
            # ========== PLOT 2: Validation Accuracy ==========
            ax2 = axes[0, 1]
            if 'epoch' in combined_df.columns and 'val_acc' in combined_df.columns:
                folds = combined_df['fold'].unique() if 'fold' in combined_df.columns else [0]
                for fold in sorted(folds)[:13]:
                    fold_data = combined_df[combined_df['fold'] == fold] if 'fold' in combined_df.columns else combined_df
                    if not fold_data.empty:
                        ax2.plot(fold_data['epoch'], fold_data['val_acc'] * 100, 
                                alpha=0.7, linewidth=2, marker='o', markersize=4,
                                label=f'Subject {int(fold)}')
                ax2.axhline(y=70, color='red', linestyle='--', linewidth=2, 
                           label='🎯 70% Target', alpha=0.7)
                ax2.axhline(y=54, color='gray', linestyle=':', linewidth=1.5, 
                           label='Baseline (54%)', alpha=0.5)
                ax2.set_xlabel('Epoch', fontsize=11)
                ax2.set_ylabel('Validation Accuracy (%)', fontsize=11)
                ax2.set_title('📈 Validation Accuracy Over Time', fontsize=13, fontweight='bold')
                ax2.legend(ncol=3, fontsize=8, loc='lower right')
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim(0, 100)
            else:
                ax2.text(0.5, 0.5, '⏳ Waiting for validation data...', 
                        ha='center', va='center', fontsize=14, transform=ax2.transAxes)
                ax2.set_xlim(0, 1)
                ax2.set_ylim(0, 1)
            
            # ========== PLOT 3: Per-Fold Best Accuracy ==========
            ax3 = axes[1, 0]
            if 'fold' in combined_df.columns and 'val_acc' in combined_df.columns:
                fold_best = combined_df.groupby('fold')['val_acc'].max() * 100
                colors = ['green' if acc >= 70 else 'orange' if acc >= 60 else 'red' 
                         for acc in fold_best.values]
                bars = ax3.bar(range(len(fold_best)), fold_best.values, 
                              color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
                ax3.axhline(y=70, color='red', linestyle='--', linewidth=2, 
                           label='🎯 70% Target', alpha=0.7)
                ax3.axhline(y=54, color='gray', linestyle=':', linewidth=1.5, 
                           label='Baseline (54%)', alpha=0.5)
                ax3.set_xlabel('Subject (Fold)', fontsize=11)
                ax3.set_ylabel('Best Accuracy (%)', fontsize=11)
                ax3.set_title('🏆 Best Accuracy Per Subject', fontsize=13, fontweight='bold')
                ax3.set_xticks(range(len(fold_best)))
                ax3.set_xticklabels([f'S{int(i)}' for i in fold_best.index], fontsize=9)
                ax3.legend(fontsize=9)
                ax3.grid(True, alpha=0.3, axis='y')
                ax3.set_ylim(0, 100)
                
                # Add value labels on bars
                for i, (bar, val) in enumerate(zip(bars, fold_best.values)):
                    ax3.text(bar.get_x() + bar.get_width()/2, val + 2, 
                            f'{val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
            else:
                ax3.text(0.5, 0.5, '⏳ Waiting for fold data...', 
                        ha='center', va='center', fontsize=14, transform=ax3.transAxes)
                ax3.set_xlim(0, 1)
                ax3.set_ylim(0, 1)
            
            # ========== PLOT 4: Progress Summary ==========
            ax4 = axes[1, 1]
            if 'val_acc' in combined_df.columns:
                completed_folds = len(combined_df['fold'].unique()) if 'fold' in combined_df.columns else 1
                total_samples = len(combined_df)
                mean_acc = combined_df['val_acc'].mean() * 100
                max_acc = combined_df['val_acc'].max() * 100
                min_acc = combined_df['val_acc'].min() * 100
                std_acc = combined_df['val_acc'].std() * 100
                
                # Calculate improvement from baseline
                improvement = mean_acc - 54.0
                improvement_emoji = "🚀" if improvement > 18 else "📈" if improvement > 10 else "⏳"
                
                status_emoji = "🎯✅" if mean_acc >= 70 else "⏳📊" if mean_acc >= 60 else "🔄💪"
                status_text = "ON TARGET!" if mean_acc >= 70 else "TRAINING..." if mean_acc >= 60 else "WARMING UP..."
                
                stats_text = f"""
╔════════════════════════════════════════╗
║     TRAINING PROGRESS DASHBOARD        ║
╚════════════════════════════════════════╝

📊 PROGRESS
   Completed Subjects: {completed_folds}/13
   Total Epochs Run: {total_samples}
   
📈 ACCURACY STATISTICS
   • Mean Accuracy:    {mean_acc:.2f}%
   • Best Accuracy:    {max_acc:.2f}%
   • Worst Accuracy:   {min_acc:.2f}%
   • Std Deviation:    {std_acc:.2f}%
   
{improvement_emoji} IMPROVEMENT vs BASELINE
   Baseline:  54.00%
   Current:   {mean_acc:.2f}%
   Gain:      +{improvement:.2f}%
   
🎯 TARGET STATUS
   Target:    70.00%
   Progress:  {mean_acc:.2f}%
   Status:    {status_emoji} {status_text}
   
⏱️  UPDATED: {time.strftime('%H:%M:%S')}
                """
                
                # Color-code background based on performance
                bg_color = 'lightgreen' if mean_acc >= 70 else 'lightyellow' if mean_acc >= 60 else 'lightcoral'
                
                ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
                        fontsize=11, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.3, 
                                 edgecolor='black', linewidth=2))
                ax4.axis('off')
            else:
                ax4.text(0.5, 0.5, '⏳ Initializing...', 
                        ha='center', va='center', fontsize=14, transform=ax4.transAxes)
                ax4.axis('off')
            
            plt.tight_layout()
            display.clear_output(wait=True)
            display.display(fig)
            
            # Console update
            if 'val_acc' in combined_df.columns:
                print(f"📊 Update {update_num+1} @ {time.strftime('%H:%M:%S')} | "
                      f"Folds: {completed_folds}/13 | "
                      f"Mean Acc: {mean_acc:.2f}% | "
                      f"Best: {max_acc:.2f}% | "
                      f"Status: {status_text}")
            
            time.sleep(refresh_seconds)
            
        except KeyboardInterrupt:
            print("\n⏹️ Monitoring stopped by user")
            break
        except Exception as e:
            print(f"⚠️ Error in update {update_num+1}: {e}")
            time.sleep(refresh_seconds)
    
    plt.ioff()
    plt.show()
    print("\n" + "="*70)
    print("✅ MONITORING COMPLETE!")
    print("="*70)

# Auto-start monitoring after training begins
# Run this cell AFTER starting Cell 6 (training)
print("\n💡 INSTRUCTIONS:")
print("1. Run Cell 6 to start training")
print("2. Then run the command below in THIS cell to start monitoring:\n")
print("monitor_training_live(")
print(f"    '{REPO_DIR}/Code-base/MocapDatasetScripting_REALLAB/results',")
print("    refresh_seconds=30")
print(")")
print("\n" + "="*70)

# Uncomment to auto-start (runs automatically after Cell 6):
# monitor_training_live(f'{REPO_DIR}/Code-base/MocapDatasetScripting_REALLAB/results', refresh_seconds=30)
