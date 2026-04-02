# Colab GPU Run (VS Code Colab Extension)

This project now supports Colab-friendly environment overrides in:
- Code-base/MocapDatasetScripting_REALLAB/scripts/training/conv1d_bigru_loso.py

Use the notebook below to run with CUDA GPU from the VS Code Google Colab extension:
- Code-base/MocapDatasetScripting_REALLAB/scripts/training/colab_gpu_train_loso.ipynb

## Steps
1. Open the notebook in VS Code.
2. Connect with the Google Colab extension and set runtime to GPU.
3. Edit the path variables in the config cell:
   - REPO_DIR
   - H5_PATH
4. Run the notebook cells top to bottom.

## Fast defaults used in notebook
- FAST_MODE=1 (higher throughput on CUDA)
- DATALOADER_WORKERS=4
- PREFETCH_FACTOR=4
- PERSISTENT_WORKERS=1
- MATMUL_PRECISION=high
- RESULT_TAG=colab_gpu

## Important
- Smoke test first by setting MAX_FOLDS=1.
- For full run, set MAX_FOLDS=0.
- Results are written to the configured RESULTS_FOLDER and zipped to RESULTS_ZIP_PATH.
