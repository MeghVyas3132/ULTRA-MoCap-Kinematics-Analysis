You are working in this repository:
- Project root: /Users/meghvyas/Desktop/research-paper
- Main script: Code-base/MocapDatasetScripting_REALLAB/scripts/training/conv1d_bigru_loso.py

Primary objective:
- Keep IMU and IMU+EMG performance at least as strong as current baseline.
- Improve EMG-only performance substantially (target major gain in both Accuracy and Macro-F1).

Required outputs:
1) Full LOSO run on all subjects.
2) CSV metrics for all modalities.
3) Automatic EMG baseline-vs-current comparison table.
4) Final summary table with mean Accuracy and mean Macro-F1 for:
   - EMG baseline (ConvBiGRU)
   - EMG current variant (LSTM-MSA)
   - IMU baseline
   - IMU+EMG baseline
5) Brief conclusion stating whether the objective was met.

Environment setup steps:
1. Configure Python environment for this workspace.
2. Ensure packages are installed: h5py, numpy, pandas, scipy, tqdm, scikit-learn, torch.

Execution plan:
1. Validate script syntax before long run.
2. Run a one-fold smoke test first:
   SMOKE_TEST_SUBJECT=subject_1 NUM_EPOCHS=1 BATCH_SIZE=64 \
   /Users/meghvyas/Desktop/research-paper/.venv/bin/python \
   Code-base/MocapDatasetScripting_REALLAB/scripts/training/conv1d_bigru_loso.py

3. Run full LOSO benchmark (all subjects, all modalities):
   /Users/meghvyas/Desktop/research-paper/.venv/bin/python \
   Code-base/MocapDatasetScripting_REALLAB/scripts/training/conv1d_bigru_loso.py

4. Confirm output files exist in:
   Code-base/MocapDatasetScripting_REALLAB/results/Results_ConvBiGRU/

Mandatory files to verify:
- Crossval_results_EMGOnly_ConvBiGRU.csv
- Crossval_results_EMGOnly_LSTM_MSA.csv
- Crossval_results_IMUOnly_ConvBiGRU.csv
- Crossval_results_IMU_EMG_ConvBiGRU.csv
- Crossval_results_combined_EMG_lstm_msa.csv
- EMG_baseline_vs_lstm_msa_by_subject.csv
- EMG_baseline_vs_lstm_msa_summary.csv

Post-run analysis requirements:
1. Compute and report means for Accuracy and Macro-F1 across folds for each modality CSV.
2. Compare IMU and IMU+EMG means against prior baseline values in this repo:
   - IMU mean accuracy around 0.946
   - IMU+EMG mean accuracy around 0.957
   Do not claim regression unless recomputed on identical fold scope.
3. Use EMG comparison summary CSV to quantify gain/loss.
4. If EMG did not improve, run one controlled iteration:
   - Keep IMU and IMU+EMG architecture unchanged.
   - Tune only EMG settings (dropout, hidden size, learning rate, epochs) and rerun EMG-only LOSO.
   - Regenerate EMG comparison CSVs.

Acceptance criteria:
- IMU and IMU+EMG performance maintained (no meaningful drop).
- EMG accuracy and EMG macro-F1 improved clearly over baseline.
- All required CSVs present and summarized.

Final deliverable format:
- Section 1: Run status and any errors.
- Section 2: Metrics table (means only) for all modalities.
- Section 3: EMG gain table (baseline vs current, with deltas).
- Section 4: Objective verdict (met / partially met / not met) with one-paragraph explanation.
