#!/bin/bash
# Show training results clearly

RESULTS_DIR=~/Desktop/research-paper/Code-base/MocapDatasetScripting_REALLAB/results

echo "📊 EMG Training Results"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Find the main results CSV
MAIN_CSV=$(find "$RESULTS_DIR" -name "*summary*" -o -name "*loso*" -o -name "*all*" 2>/dev/null | grep -v "fold" | head -1)

if [ -z "$MAIN_CSV" ]; then
    echo "⏳ Training in progress..."
    echo ""
    echo "Checking individual fold files..."
    find "$RESULTS_DIR" -name "*.csv" -type f 2>/dev/null | head -5 | while read f; do
        echo "  - $(basename $f)"
    done
else
    echo "Main results: $(basename $MAIN_CSV)"
    echo ""
    cat "$MAIN_CSV"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
