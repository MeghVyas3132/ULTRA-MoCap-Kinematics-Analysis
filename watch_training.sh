#!/bin/bash
# Watch live training progress

echo "🔄 Watching training progress..."
echo "Press Ctrl+C to stop watching (training continues)"
echo ""

while true; do
    clear
    echo "📊 EMG Training Monitor - $(date '+%H:%M:%S')"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # Find newest CSV
    LATEST=$(find ~/Desktop/research-paper/Code-base/MocapDatasetScripting_REALLAB/results -name "*.csv" -type f -exec ls -t {} + 2>/dev/null | grep -v baseline | grep -v summary | head -1)
    
    if [ ! -z "$LATEST" ]; then
        echo "Latest file: $(basename $LATEST)"
        echo ""
        tail -10 "$LATEST"
    else
        echo "⏳ Waiting for results..."
        echo ""
        echo "Training is running - results will appear when fold completes"
    fi
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Auto-refreshing every 10 seconds..."
    
    sleep 10
done
