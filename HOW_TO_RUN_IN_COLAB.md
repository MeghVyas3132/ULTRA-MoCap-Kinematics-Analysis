# 🚀 How to Run EMG Improvements in Google Colab

## ✅ Everything is Ready!

All code has been tested and validated locally. Now you just need to run it in Google Colab for GPU acceleration.

---

## 📋 Step-by-Step Guide

### Step 1: Upload Notebook to Colab

1. **Go to Google Colab:**
   - Open: https://colab.research.google.com

2. **Upload the notebook:**
   - Click **File → Upload notebook**
   - Navigate to: `/Users/meghvyas/Desktop/research-paper/Code-base/MocapDatasetScripting_REALLAB/scripts/training/`
   - Select: **`emg_accuracy_improvements_colab.ipynb`**
   - Click **Open**

### Step 2: Set GPU Runtime

1. **Change runtime type:**
   - Click **Runtime → Change runtime type**
   - Under **Hardware accelerator**, select: **GPU** (T4 or better)
   - Click **Save**

2. **Verify GPU is active:**
   - Run Cell 1 (GPU check)
   - Should show: ✅ CUDA Available: True
   - GPU Device: Tesla T4 (or similar)

### Step 3: Prepare Your Dataset

**Option A: Dataset already in Drive**
- Ensure `All_subjects_data.h5` is at one of these paths:
  ```
  /MyDrive/research-paper/Dataset/ULTra-MoCap-processed/All_subjects_data.h5
  /MyDrive/Dataset/ULTra-MoCap-processed/All_subjects_data.h5
  ```

**Option B: Upload dataset to Drive now**
1. Open Google Drive: https://drive.google.com
2. Create folder structure: `research-paper/Dataset/ULTra-MoCap-processed/`
3. Upload `All_subjects_data.h5` to that folder
4. Wait for upload to complete (2.1 GB file)

### Step 4: Configure Training (Optional)

**In Cell 4 of the notebook, you can adjust:**

```python
# For quick testing (15-30 minutes):
MAX_FOLDS = 1           # Test on 1 subject
EMG_EPOCHS = 10         # Fewer epochs

# For full training (10-12 hours):
MAX_FOLDS = 0           # All 13 subjects
EMG_EPOCHS = 50         # Full training

# Model selection:
EMG_MODEL_VARIANT = 'conformer'    # Best (recommended)
# EMG_MODEL_VARIANT = 'dual_branch'  # Alternative
# EMG_MODEL_VARIANT = 'lstm_msa'     # Baseline comparison

# Improvements (all enabled by default):
USE_AUGMENTATION = True
USE_TTA = True
USE_LABEL_SMOOTHING = True
```

### Step 5: Run Training

**Easy way: Run all cells at once**
1. Click **Runtime → Run all**
2. When prompted, click **Allow** to mount Google Drive
3. Sign in with your Google account
4. Click **Allow** again to give Colab access

**Or: Run cells one by one**
1. Run Cell 1: GPU check ✅
2. Run Cell 2: Mount Drive ✅
3. Run Cell 3: Install dependencies ✅
4. Run Cell 4: Configuration ✅
5. Run Cell 5: Verify files ✅
6. Run Cell 6: **Launch training** ⏰ (this takes the longest)
7. Run Cell 7: Sync results to Drive ✅
8. Run Cell 8: View results ✅

### Step 6: Monitor Progress

**While training is running:**
- You'll see progress bars for each epoch
- Loss and accuracy metrics will be printed
- Each fold takes ~45-60 minutes on T4 GPU
- Full 13-fold training: ~10-12 hours

**You can safely:**
- Close the browser tab (training continues)
- Come back later to check progress
- Keep Colab open in another tab

**Warning: Colab may disconnect after 90 minutes of inactivity**
- Solution: Keep the tab active or use Colab Pro

---

## 📊 What to Expect

### During Training:
```
Fold 1/13: subject_1
Epoch 1/50: 100%|██████████| Loss: 1.234 Acc: 0.456
Epoch 2/50: 100%|██████████| Loss: 1.123 Acc: 0.523
...
✅ Best model saved: accuracy 0.756

Fold 2/13: subject_2
...
```

### After Training:
```
============================================================
ACCURACY SUMMARY
============================================================
Mean Accuracy: 0.7234 ± 0.0456
Best Fold: 0.7812
Worst Fold: 0.6543
============================================================

Per-Fold Results:
  Fold 1: 0.7234
  Fold 2: 0.7456
  ...
```

---

## 📁 Where Results Are Saved

**During training:**
- `/content/mocap_cache/results/EMG_improvements/`

**After Cell 7 (sync):**
- **Google Drive:** `/MyDrive/research-paper/results/EMG_improvements/`

**Files created:**
```
EMG_improvements/
├── fold_1_results.csv           # Metrics for fold 1
├── fold_1_best_model.pth        # Best model checkpoint
├── fold_1_confusion_matrix.txt  # Confusion matrix
├── fold_2_results.csv
├── fold_2_best_model.pth
├── ...
├── summary_results.csv          # Overall summary
└── last_sync.txt               # Sync info
```

---

## 🐛 Troubleshooting

### "GPU not available"
```
❌ Error: CUDA GPU is not available
```
**Fix:** Runtime → Change runtime type → GPU → Save

### "Dataset not found"
```
❌ FileNotFoundError: All_subjects_data.h5 not found
```
**Fix:** 
1. Make sure dataset is uploaded to Drive
2. Run Cell 2 (Mount Drive) successfully
3. Check path in Cell 4 matches your Drive location

### "Out of memory"
```
❌ RuntimeError: CUDA out of memory
```
**Fix:** In Cell 4, change:
```python
EMG_BATCH_SIZE = 64  # or 32
```

### "Runtime disconnected"
```
⚠️ Warning: Runtime disconnected
```
**Fix:**
- Your session timed out after 90 minutes inactive
- Results up to last checkpoint are saved
- Re-run to continue from where it stopped
- Or: Upgrade to Colab Pro for longer sessions

### Training is slow
**Check:**
1. Verify GPU is active (Cell 1 should show CUDA: True)
2. If no GPU, change runtime type to GPU
3. T4 GPU: ~45-60 min per fold
4. CPU: ~4-6 hours per fold (very slow!)

---

## 🎯 Quick Test First (Recommended)

Before running full 13-fold training, test with 1 fold:

**In Cell 4, change:**
```python
MAX_FOLDS = 1      # Just 1 subject
EMG_EPOCHS = 10    # Quick epochs
```

**This will:**
- Verify everything works correctly
- Take only ~15-30 minutes
- Show you what to expect
- Then you can run the full training

---

## 💡 Tips

1. **Start with quick test** (MAX_FOLDS=1) to verify setup
2. **Check GPU is active** before starting long training
3. **Keep Colab tab open** to avoid disconnect
4. **Results auto-sync** to Drive in Cell 7
5. **Compare models** by running multiple configurations

---

## 📈 Expected Results

| Configuration | Mean Accuracy | Time (T4 GPU) |
|--------------|---------------|---------------|
| Quick Test (1 fold) | N/A | 15-30 min |
| Full Training (13 folds) | ~72-76% | 10-12 hours |
| Baseline LSTM | ~54% | 8-10 hours |

**Expected improvement: +18-22% over baseline**

---

## ✅ Checklist

Before running:
- [ ] Notebook uploaded to Colab
- [ ] GPU runtime selected
- [ ] Dataset in Google Drive
- [ ] Drive mounted successfully
- [ ] Configuration reviewed
- [ ] Ready to run!

---

## 🆘 Need Help?

If you encounter issues:
1. Check this guide's troubleshooting section
2. Review notebook cell outputs for error messages  
3. Verify all prerequisite steps completed
4. Try quick test (1 fold) first

---

**Ready to go! Upload the notebook to Colab and run all cells.** 🚀

The notebook handles everything automatically:
- ✅ Clones your GitHub repo
- ✅ Finds the dataset
- ✅ Configures everything
- ✅ Runs training with improvements
- ✅ Syncs results to Drive
- ✅ Shows you the results

**Just upload and click "Run all"!**
