#!/bin/bash
# Simple script to check accuracy per fold

RESULTS_DIR=~/Desktop/research-paper/Code-base/MocapDatasetScripting_REALLAB/results

echo "📊 EMG Training Progress"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Count CSV files
CSV_COUNT=$(find "$RESULTS_DIR" -name "*.csv" -type f 2>/dev/null | wc -l | tr -d ' ')

if [ "$CSV_COUNT" -eq 0 ]; then
    echo "⏳ No results yet - fold 1 still training..."
    echo ""
    echo "Results will appear when first fold completes."
    exit 0
fi

echo "Completed folds: $CSV_COUNT"
echo ""

# Show each fold's accuracy
find "$RESULTS_DIR" -name "*.csv" -type f 2>/dev/null | while read file; do
    if [ -f "$file" ]; then
        # Get last line and extract accuracy
        ACC=$(tail -1 "$file" | cut -d',' -f3)
        FOLD=$(tail -1 "$file" | cut -d',' -f1)
        
        if [ ! -z "$ACC" ] && [ ! -z "$FOLD" ]; then
            ACC_PCT=$(echo "$ACC * 100" | bc -l 2>/dev/null | cut -d'.' -f1)
            echo "Fold $FOLD: ${ACC_PCT}%"
        fi
    fi
done | sort -t: -k1 -n

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
