# 🖥️ COLAB TERMINAL COMMANDS - Monitor Training in Real-Time

**Run these commands in Colab's terminal WHILE training is running**  
Training will NOT be interrupted!

---

## 🚀 Quick Start - Just Copy & Paste This:

### In Colab, create a NEW CODE CELL and run:

```bash
%%bash
# This runs in background and shows live updates every 30 seconds

RESULTS_DIR="/content/ULTRA-MoCap-Kinematics-Analysis/Code-base/MocapDatasetScripting_REALLAB/results"

while true; do
    clear
    echo "🔄 Training Monitor - $(date '+%H:%M:%S')"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Check training process
    if ps aux | grep python | grep -E "(conv1d|run_emg)" | grep -v grep > /dev/null; then
        echo "✅ Training RUNNING"
    else
        echo "⏸️  Training not detected"
    fi
    
    # GPU status
    echo ""
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
    awk -F, '{printf "🎮 GPU: %s%% | VRAM: %s/%s MB\n", $1, $2, $3}'
    
    # Latest results
    echo ""
    LATEST=$(find "$RESULTS_DIR" -name "*.csv" -type f -exec ls -t {} + 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        echo "📊 Latest Results from: $(basename "$LATEST")"
        echo ""
        tail -6 "$LATEST" | head -1  # Header
        tail -5 "$LATEST"             # Last 5 rows
    else
        echo "⏳ Waiting for results..."
    fi
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    sleep 30
done
```

**Press STOP button in Colab to end monitoring (training continues!)**

---

## 📊 Alternative Commands

### 1. **One-Time Status Check**
```bash
!echo "Training Status:" && \
ps aux | grep python | grep -E "(conv1d|run_emg)" | grep -v grep | head -1 && \
echo "" && \
echo "GPU Usage:" && \
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader && \
echo "" && \
echo "Latest Results:" && \
find /content/ULTRA-MoCap-Kinematics-Analysis/Code-base/MocapDatasetScripting_REALLAB/results/ -name "*.csv" -exec ls -lt {} + | head -1 | awk '{print $NF}' | xargs tail -5 2>/dev/null || echo "No results yet"
```

### 2. **Watch Results Directory**
```bash
!watch -n 20 'ls -lht /content/ULTRA-MoCap-Kinematics-Analysis/Code-base/MocapDatasetScripting_REALLAB/results/ | head -10'
```

### 3. **Show Latest Accuracy**
```bash
!LATEST=$(find /content/ULTRA-MoCap-Kinematics-Analysis/Code-base/MocapDatasetScripting_REALLAB/results/ -name "*.csv" -exec ls -t {} + | head -1) && \
if [ -n "$LATEST" ]; then \
    echo "Latest Results:" && tail -5 "$LATEST"; \
else \
    echo "No results yet"; \
fi
```

### 4. **GPU Monitoring Only**
```bash
!watch -n 10 nvidia-smi
```

### 5. **Count Completed Folds**
```bash
!echo "Completed CSV files:" && \
find /content/ULTRA-MoCap-Kinematics-Analysis/Code-base/MocapDatasetScripting_REALLAB/results/ -name "*.csv" -type f | wc -l
```

---

## 🎯 Best Practice - Two Cell Approach:

### Cell 1: Start Training
```python
# Your training cell (Cell 6)
# This runs and keeps going...
```

### Cell 2: Monitor Progress (Run in parallel)
```bash
%%bash
# Monitor script (paste the Quick Start script above)
# This shows updates every 30 seconds
# Press STOP to end monitoring (training continues)
```

---

## 📋 What You'll See:

```
🔄 Training Monitor - 14:23:45
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Training RUNNING

🎮 GPU: 98% | VRAM: 14256/15360 MB

📊 Latest Results from: fold_3_results.csv

epoch,train_loss,val_acc,val_f1
46,0.2134,0.7245,0.7123
47,0.2089,0.7312,0.7201
48,0.2045,0.7389,0.7278
49,0.2001,0.7456,0.7355
50,0.1967,0.7523,0.7432

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Next update in 30 seconds...
```

---

## ✅ Key Points:

- ✅ **Safe**: Won't interrupt training
- ✅ **Live**: Updates every 30 seconds
- ✅ **Minimal**: Low overhead, won't slow training
- ✅ **Parallel**: Run in separate cell while training continues
- ✅ **Flexible**: Stop/start monitoring anytime

---

## 🔧 Troubleshooting:

**Q: "Nothing showing up?"**  
A: Training may still be in setup phase. Wait 2-3 minutes for first results.

**Q: "How do I stop monitoring?"**  
A: Press the STOP button in Colab cell. Training continues unaffected.

**Q: "Can I run multiple monitors?"**  
A: Yes! Run different commands in different cells.

**Q: "Results not updating?"**  
A: Check if training is still running with `!ps aux | grep python`

---

## 📦 Files Location:

Results are saved to:
```
/content/ULTRA-MoCap-Kinematics-Analysis/Code-base/MocapDatasetScripting_REALLAB/results/
```

Each fold creates a CSV file with epoch-by-epoch results.

---

**TL;DR - Copy the Quick Start bash script into a new cell and run it!** 🚀
