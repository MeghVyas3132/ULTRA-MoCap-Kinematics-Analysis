# 🚀 Quick Reference: Running EMG Improvements

## Colab (Fastest - Recommended)

```
1. Go to: https://colab.research.google.com
2. Upload: emg_accuracy_improvements_colab.ipynb
3. Runtime → Change runtime type → GPU
4. Runtime → Run all
5. Done! Results in ~10-12 hours
```

## Local Mac (Slower)

```bash
cd /Users/meghvyas/Desktop/research-paper
source .venv/bin/activate
python Code-base/MocapDatasetScripting_REALLAB/scripts/training/run_emg_test_local.py
```

## Files You Need

**To run in Colab:**
- `emg_accuracy_improvements_colab.ipynb` ⭐
- Dataset in Google Drive: `/MyDrive/research-paper/Dataset/ULTra-MoCap-processed/All_subjects_data.h5`

**For reference:**
- `HOW_TO_RUN_IN_COLAB.md` - Detailed guide
- `COLAB_DEPLOYMENT_SUMMARY.md` - Technical details

## Quick Configuration

**Edit Cell 4 in notebook:**

```python
# Quick test (15-30 min):
MAX_FOLDS = 1
EMG_EPOCHS = 10

# Full training (10-12 hrs):
MAX_FOLDS = 0
EMG_EPOCHS = 50

# Model variants:
EMG_MODEL_VARIANT = 'conformer'    # Best
EMG_MODEL_VARIANT = 'dual_branch'  # Alternative
EMG_MODEL_VARIANT = 'lstm_msa'     # Baseline
```

## Expected Results

| Metric | Value |
|--------|-------|
| Baseline accuracy | ~54% |
| Improved accuracy | ~72-76% |
| Accuracy gain | +18-22% |
| Training time (Colab T4) | 10-12 hours |
| Training time (Mac MPS) | 24-30 hours |

## Where Results Are Saved

**Google Drive:**
```
/MyDrive/research-paper/results/EMG_improvements/
├── fold_X_results.csv
├── fold_X_best_model.pth
├── summary_results.csv
└── last_sync.txt
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| No GPU | Runtime → Change runtime type → GPU |
| Dataset not found | Check Drive path, ensure mounted |
| Out of memory | Reduce `EMG_BATCH_SIZE` to 64 or 32 |
| Runtime disconnect | Keep tab open, or use Colab Pro |

## Need Help?

📖 Read `HOW_TO_RUN_IN_COLAB.md` for full guide

---

**TL;DR: Upload notebook to Colab, set GPU, click "Run all"** ✅
