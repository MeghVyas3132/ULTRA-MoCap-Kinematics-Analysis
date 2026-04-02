#!/bin/bash
# Comprehensive training progress check

RESULTS_CSV=~/Desktop/research-paper/Code-base/MocapDatasetScripting_REALLAB/results/Results_ConvBiGRU/Crossval_results_combined_EMG_conformer.csv

echo "📊 EMG CONFORMER TRAINING - PROGRESS REPORT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ -f "$RESULTS_CSV" ]; then
    echo "Completed Folds:"
    echo ""
    
    # Show individual fold results
    awk -F',' 'NR>1 {
        printf "  Fold %2s (%s): %5.1f%% accuracy | F1: %.3f\n", 
        $1, $2, $4*100, $7
    }' "$RESULTS_CSV"
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # Calculate average
    AVG_ACC=$(awk -F',' 'NR>1 {sum+=$4; count++} END {if(count>0) printf "%.1f", (sum/count)*100}' "$RESULTS_CSV")
    AVG_F1=$(awk -F',' 'NR>1 {sum+=$7; count++} END {if(count>0) printf "%.3f", sum/count}' "$RESULTS_CSV")
    COMPLETED=$(awk -F',' 'NR>1 {count++} END {print count}' "$RESULTS_CSV")
    
    echo "Summary:"
    echo "  ✓ Completed: $COMPLETED/13 folds"
    echo "  📈 Average Accuracy: ${AVG_ACC}%"
    echo "  📈 Average F1 Score: ${AVG_F1}"
    echo ""
    
    # Compare to baseline
    BASELINE=54.0
    if (( $(echo "$AVG_ACC > $BASELINE" | bc -l) )); then
        DIFF=$(echo "$AVG_ACC - $BASELINE" | bc -l)
        printf "  🎯 Performance: +%.1f%% vs baseline (54%%)\n" "$DIFF"
    else
        DIFF=$(echo "$BASELINE - $AVG_ACC" | bc -l)
        printf "  ⚠️  Performance: -%.1f%% vs baseline (54%%)\n" "$DIFF"
    fi
    
else
    echo "⏳ No results yet - training just started"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
