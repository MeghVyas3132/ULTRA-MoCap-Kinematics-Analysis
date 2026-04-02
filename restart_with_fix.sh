#!/bin/bash
# Restart training with correct model (EMGConformer)

echo "🔄 RESTARTING WITH CORRECT MODEL"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "This will use EMGConformer (1.55M params) instead of"
echo "the baseline EMGConvBiGRUModel (~500K params)"
echo ""
echo "Expected accuracy: 72-76% (vs 54% baseline)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "⚠️  FIRST: Stop the current training in the other terminal"
echo "    Press Ctrl+C in the terminal running training"
echo ""
read -p "Press ENTER when you've stopped the old training... "
echo ""
echo "Starting CORRECT training..."
echo ""

cd ~/Desktop/research-paper
python3 run_local_training.py
