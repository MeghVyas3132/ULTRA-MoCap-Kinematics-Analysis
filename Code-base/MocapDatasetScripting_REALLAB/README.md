# ULTra MoCap Processing

This folder contains training, preprocessing, visualization, and experiment artifacts for ULTRA-MoCap movement classification.

## Project Layout

- `scripts/training/conv1d_bigru_loso.py`
	- Main LOSO training script.
	- Supports `emg`, `imu`, `imu_emg` modalities.
	- Includes `EMG_MODEL_VARIANT` (`convbigru` or `lstm_msa`).

- `scripts/training/personalized_validation.py`
	- Personalized/adaptation validation workflow.

- `scripts/visualization/visualize_subject_distribution.py`
	- Subject-level distribution plots.

- `scripts/visualization/visualize_speed_distributions.py`
	- Speed distribution plots.

- `processing/`
	- Data preparation scripts/notebooks.

- `datasets/`
	- Pre-sharded LOSO train/test windows.

- `results/Results_ConvBiGRU/`
	- Per-fold model checkpoints and benchmark CSV outputs.

- `docs/reports/ConvBiGRU_Summary_Report.pdf`
	- Generated report artifact.

- `media/`
	- Figures and GIFs used for documentation.

## Backward Compatibility

The original root-level script names are preserved as launcher wrappers:

- `Conv1D-Bi-GRU_Implementation.py`
- `Personalized_validation.py`
- `visualize_subject_distribution.py`
- `visualize_speed_distributions.py`

They forward execution to the new files in `scripts/`.

## Run Commands

From workspace root (`/Users/meghvyas/Desktop/research-paper`):

```bash
# Full LOSO (all modalities, default settings)
/Users/meghvyas/Desktop/research-paper/.venv/bin/python \
	Code-base/MocapDatasetScripting_REALLAB/scripts/training/conv1d_bigru_loso.py

# Quick one-fold smoke test
SMOKE_TEST_SUBJECT=subject_1 NUM_EPOCHS=1 BATCH_SIZE=64 \
	/Users/meghvyas/Desktop/research-paper/.venv/bin/python \
	Code-base/MocapDatasetScripting_REALLAB/scripts/training/conv1d_bigru_loso.py

# EMG-only smoke test for fast iteration
SMOKE_TEST_SUBJECT=subject_1 MODALITIES=emg NUM_EPOCHS=1 BATCH_SIZE=64 \
	/Users/meghvyas/Desktop/research-paper/.venv/bin/python \
	Code-base/MocapDatasetScripting_REALLAB/scripts/training/conv1d_bigru_loso.py
```

## Benchmark Outputs

Key outputs are written to `results/Results_ConvBiGRU/`:

- `Crossval_results_EMGOnly_ConvBiGRU.csv` (baseline EMG)
- `Crossval_results_EMGOnly_LSTM_MSA.csv` (current EMG variant)
- `Crossval_results_IMUOnly_ConvBiGRU.csv`
- `Crossval_results_IMU_EMG_ConvBiGRU.csv`
- `Crossval_results_combined_EMG_lstm_msa.csv`
- `EMG_baseline_vs_lstm_msa_by_subject.csv`
- `EMG_baseline_vs_lstm_msa_summary.csv`

The EMG comparison CSVs are automatically generated at the end of the training script.
