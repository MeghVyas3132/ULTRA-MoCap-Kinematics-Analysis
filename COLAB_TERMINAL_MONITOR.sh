#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════╗
# ║          COLAB TERMINAL COMMANDS - MONITOR TRAINING              ║
# ║              Run these WHILE training is running                 ║
# ╚══════════════════════════════════════════════════════════════════╝

# ============================================================
# OPTION 1: WATCH TRAINING OUTPUT (RECOMMENDED)
# ============================================================
# This shows the STDOUT/STDERR from training in real-time
# If training outputs to console, you'll see it here

# For Python script with unbuffered output:
ps aux | grep python | grep -v grep | awk '{print $2}' | head -1 | xargs -I {} tail -f /proc/{}/fd/1 2>&1

# Simpler version - just watch the process:
watch -n 5 'ps aux | grep -E "(python|training)" | grep -v grep'


# ============================================================
# OPTION 2: MONITOR RESULTS FILES (BEST FOR COLAB)
# ============================================================
# Watch results directory for new files and changes

# See latest results file updates:
watch -n 10 'ls -lht /content/ULTRA-MoCap-Kinematics-Analysis/Code-base/MocapDatasetScripting_REALLAB/results/ | head -20'

# Auto-refresh and show latest CSV content:
watch -n 30 'find /content/ULTRA-MoCap-Kinematics-Analysis/Code-base/MocapDatasetScripting_REALLAB/results/ -name "*.csv" -type f -exec ls -lt {} + | head -1 | awk "{print \$NF}" | xargs tail -20'


# ============================================================
# OPTION 3: LIVE PROGRESS MONITOR (COPY-PASTE THIS)
# ============================================================
cat << 'MONITOR_SCRIPT' > /tmp/monitor_training.sh
#!/bin/bash
# Real-time training monitor for Colab

RESULTS_DIR="/content/ULTRA-MoCap-Kinematics-Analysis/Code-base/MocapDatasetScripting_REALLAB/results"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║            EMG TRAINING MONITOR - LIVE UPDATES                   ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

while true; do
    clear
    echo "🔄 Updated: $(date '+%H:%M:%S')"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Check if training is running
    TRAINING_PID=$(ps aux | grep python | grep -E "(conv1d|run_emg|training)" | grep -v grep | awk '{print $2}' | head -1)
    if [ -n "$TRAINING_PID" ]; then
        echo "✅ Training RUNNING (PID: $TRAINING_PID)"
        CPU=$(ps -p $TRAINING_PID -o %cpu --no-headers 2>/dev/null | xargs)
        MEM=$(ps -p $TRAINING_PID -o %mem --no-headers 2>/dev/null | xargs)
        echo "   CPU: ${CPU}% | Memory: ${MEM}%"
    else
        echo "⏸️  No training process detected"
    fi
    
    echo ""
    echo "📊 GPU STATUS:"
    nvidia-smi --query-gpu=gpu_name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | \
    awk -F, '{printf "   GPU: %s\n   Temp: %s°C | GPU Util: %s%% | Mem Util: %s%%\n   VRAM: %s MB / %s MB\n", $1, $2, $3, $4, $5, $6}' || echo "   GPU info not available"
    
    echo ""
    echo "📁 RESULTS FILES:"
    if [ -d "$RESULTS_DIR" ]; then
        FILE_COUNT=$(find "$RESULTS_DIR" -name "*.csv" -type f 2>/dev/null | wc -l)
        echo "   Total CSV files: $FILE_COUNT"
        
        LATEST_FILE=$(find "$RESULTS_DIR" -name "*.csv" -type f -exec ls -t {} + 2>/dev/null | head -1)
        if [ -n "$LATEST_FILE" ]; then
            echo "   Latest: $(basename "$LATEST_FILE")"
            echo "   Modified: $(stat -c %y "$LATEST_FILE" 2>/dev/null | cut -d' ' -f1-2 || stat -f "%Sm" "$LATEST_FILE" 2>/dev/null)"
            
            echo ""
            echo "📈 LATEST RESULTS (last 5 lines):"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            tail -6 "$LATEST_FILE" | head -1  # Header
            tail -5 "$LATEST_FILE"
            
            # Extract accuracy if available
            if grep -q "val_acc" "$LATEST_FILE" 2>/dev/null; then
                BEST_ACC=$(tail -n +2 "$LATEST_FILE" | cut -d',' -f3 2>/dev/null | sort -rn | head -1 2>/dev/null)
                LATEST_ACC=$(tail -1 "$LATEST_FILE" | cut -d',' -f3 2>/dev/null)
                if [ -n "$BEST_ACC" ] && [ -n "$LATEST_ACC" ]; then
                    BEST_PCT=$(echo "$BEST_ACC * 100" | bc -l 2>/dev/null | cut -d'.' -f1)
                    LATEST_PCT=$(echo "$LATEST_ACC * 100" | bc -l 2>/dev/null | cut -d'.' -f1)
                    echo ""
                    echo "🎯 ACCURACY STATUS:"
                    echo "   Latest: ${LATEST_PCT}% | Best: ${BEST_PCT}%"
                    if [ "$BEST_PCT" -ge 70 ] 2>/dev/null; then
                        echo "   ✅ TARGET REACHED! (70%+)"
                    elif [ "$BEST_PCT" -ge 60 ] 2>/dev/null; then
                        echo "   📈 Good progress! Approaching target..."
                    fi
                fi
            fi
        fi
    else
        echo "   Results directory not found"
    fi
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Press Ctrl+C to stop monitoring (training will continue)"
    echo ""
    
    sleep 30
done
MONITOR_SCRIPT

chmod +x /tmp/monitor_training.sh
/tmp/monitor_training.sh


# ============================================================
# OPTION 4: QUICK STATUS CHECK (ONE-TIME)
# ============================================================
cat << 'STATUS_SCRIPT' > /tmp/quick_status.sh
#!/bin/bash
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                    QUICK TRAINING STATUS                         ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Training process
if ps aux | grep python | grep -E "(conv1d|run_emg|training)" | grep -v grep > /dev/null; then
    echo "✅ Training is RUNNING"
    ps aux | grep python | grep -E "(conv1d|run_emg)" | grep -v grep | awk '{print "   PID: "$2" | CPU: "$3"% | Mem: "$4"%"}'
else
    echo "❌ Training NOT running"
fi

echo ""

# GPU usage
echo "🎮 GPU Status:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
awk -F, '{printf "   GPU Utilization: %s%%\n   VRAM Used: %s MB / %s MB\n", $1, $2, $3}'

echo ""

# Results
RESULTS_DIR="/content/ULTRA-MoCap-Kinematics-Analysis/Code-base/MocapDatasetScripting_REALLAB/results"
if [ -d "$RESULTS_DIR" ]; then
    CSV_COUNT=$(find "$RESULTS_DIR" -name "*.csv" -type f | wc -l)
    echo "📊 Results: $CSV_COUNT CSV files found"
    
    LATEST=$(find "$RESULTS_DIR" -name "*.csv" -type f -exec ls -t {} + | head -1)
    if [ -n "$LATEST" ]; then
        echo "   Latest: $(basename "$LATEST")"
        echo ""
        echo "Last 3 lines:"
        tail -3 "$LATEST"
    fi
else
    echo "📊 No results yet"
fi
STATUS_SCRIPT

chmod +x /tmp/quick_status.sh
/tmp/quick_status.sh


# ============================================================
# OPTION 5: WATCH SPECIFIC METRICS
# ============================================================
# Monitor accuracy in real-time
watch -n 20 -d 'find /content/ULTRA-MoCap-Kinematics-Analysis/Code-base/MocapDatasetScripting_REALLAB/results/ -name "*.csv" -type f -exec tail -1 {} + | grep -oP "(?<=,)[0-9.]+(?=,)" | sort -rn | head -5 | awk "{print \"Accuracy: \" \$1*100 \"%\"}"'


# ============================================================
# OPTION 6: COMPACT MONITOR (MINIMAL OUTPUT)
# ============================================================
watch -n 15 'clear; date; echo ""; ps aux | grep python | grep -v grep | grep -E "(training|conv1d|run_emg)" || echo "No training running"; echo ""; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; echo ""; ls -lht /content/ULTRA-MoCap-Kinematics-Analysis/Code-base/MocapDatasetScripting_REALLAB/results/ 2>/dev/null | head -5 || echo "No results yet"'
