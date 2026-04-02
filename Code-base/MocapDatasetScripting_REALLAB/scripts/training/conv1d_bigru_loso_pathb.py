#!/usr/bin/env python3
"""
EMG-IMPROVED LOSO WITH PATH B (Conformer + All 7 Key Improvements)
Runs full 13-subject LOSO with:
  1. Subject-wise normalization
  2. Frequency-domain EMG features
  3. AdamW + OneCycleLR scheduler
  4. Conformer architecture
  5. Label smoothing + gradient clipping
  6. EMG augmentation
  7. Test-time augmentation

Deploy and run to get improved accuracy baseline.
"""

import os
import sys
import re
import csv
import math
import random
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from scipy.signal import butter, filtfilt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

# Import from existing script
sys.path.insert(0, os.path.dirname(__file__))

# Setup
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
g = torch.Generator().manual_seed(SEED)
device = torch.device("cpu")  # Use CPU since GPU not always available

print(f"\nDevice: {device}\n")

# Import original components
from conv1d_bigru_loso import (
    Config, DataSharder, ImuJointPairDataset, MOVEMENT_TYPES, MOVEMENT_TYPE_MAP,
    preprocess_imu_tensor, build_class_weights_for_subset, evaluate_model,
    config, all_subjects
)

# Import improvements
from emg_improvements import (
    SubjectNormalizer,
    preprocess_emg_enhanced,
    emg_combined_features,
    EMGAugmenter,
    EMGConformer,
    TTAWrapper,
    create_loss_with_options,
    create_optimizer_and_scheduler,
)

print("="*80)
print("EMG-IMPROVED LOSO TRAINING - PATH B (Conformer + 7 Improvements)")
print("="*80)

# Configuration
num_epochs = 30
patience = 6
batch_size = 128
window_length = 200
window_overlap = 0

num_classes = len(MOVEMENT_TYPES)
num_emg_channels = len(config.channels_emg)
num_imu_channels = len(config.channels_imu_acc) + len(config.channels_imu_gyr)

# Improved EMG settings
EMG_MODEL_VARIANT = "conformer"  # KEY CHANGE: Use Conformer
EMG_D_MODEL = 128
EMG_DROPOUT = 0.2
EMG_LR = 1e-3
EMG_WEIGHT_DECAY = 0.01
EMG_USE_CLASS_WEIGHTS = True
USE_AUGMENTATION = True  # KEY: Enable augmentation
USE_TTA = True  # KEY: Enable test-time augmentation

results_folder = "Code-base/MocapDatasetScripting_REALLAB/results/Results_EMG_PathB_Improved"
os.makedirs(results_folder, exist_ok=True)

csv_emg = os.path.join(results_folder, "Crossval_results_EMGOnly_Conformer.csv")
rows_emg = []

print(f"\nEMG Training Configuration:")
print(f"  Model variant:    {EMG_MODEL_VARIANT}")
print(f"  D model:          {EMG_D_MODEL}")
print(f"  Dropout:          {EMG_DROPOUT}")
print(f"  Learning rate:    {EMG_LR}")
print(f"  Weight decay:     {EMG_WEIGHT_DECAY}")
print(f"  Augmentation:     {USE_AUGMENTATION}")
print(f"  TTA:              {USE_TTA}")
print(f"  Results folder:   {results_folder}\n")

# LOSO Loop
for i, test_subject in enumerate(all_subjects):
    print(f"\n{'='*80}")
    print(f"Fold {i+1}/{len(all_subjects)} | Test Subject: {test_subject}")
    print(f"{'='*80}")

    train_subjects = [s for s in all_subjects if s != test_subject]

    # Load datasets
    full_train_dataset = ImuJointPairDataset(
        config, train_subjects, window_length, window_overlap, split="train",
    )
    test_dataset = ImuJointPairDataset(
        config, [test_subject], window_length, 0, split="test",
    )

    val_size = int(0.2 * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    train_idx, val_idx = random_split(range(len(full_train_dataset)), [train_size, val_size], generator=g)

    train_dataset = Subset(full_train_dataset, train_idx.indices)
    val_dataset = Subset(full_train_dataset, val_idx.indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=False)

    # Build Conformer model
    model = EMGConformer(
        n_channels=num_emg_channels,
        n_classes=num_classes,
        d_model=EMG_D_MODEL,
        n_blocks=4,
        n_heads=4,
        dropout=EMG_DROPOUT,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: Conformer | Parameters: {n_params:,}")

    # Setup optimized training
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, len(train_loader), num_epochs,
        lr=EMG_LR, weight_decay=EMG_WEIGHT_DECAY,
        use_onecycle=True
    )

    class_weights = build_class_weights_for_subset(train_dataset, num_classes, device) if EMG_USE_CLASS_WEIGHTS else None
    criterion = create_loss_with_options(
        num_classes, use_class_weights=EMG_USE_CLASS_WEIGHTS,
        class_weights=class_weights, label_smoothing=0.1, device=device
    )

    # Initialize augmentation and normalization
    augmenter = EMGAugmenter(prob_noise=0.5, prob_scale=0.5, prob_warp=0.3)
    normalizer = SubjectNormalizer()

    best_val_loss = float("inf")
    patience_counter = 0
    normalizer_fitted = False

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            _, _, _, emg, labels = batch

            # Improvement 1: Subject-wise normalization
            emg_proc = preprocess_emg_enhanced(
                emg, subject_normalizer=normalizer,
                is_training=(not normalizer_fitted)
            )
            normalizer_fitted = True

            # Improvement 6: Data augmentation
            if USE_AUGMENTATION and np.random.rand() < 0.7:
                emg_np = emg.numpy()
                emg_aug = augmenter(emg_np)
                emg_proc = torch.tensor(emg_aug, dtype=torch.float32)

            emg_proc = emg_proc.to(device)
            labels = labels.to(device).argmax(dim=1)

            outputs = model(emg_proc)
            loss = criterion(outputs, labels)

            # Improvement 5: Gradient clipping
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Improvement 3: OneCycleLR scheduling (step per batch)
            scheduler.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / max(len(train_loader), 1)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                _, _, _, emg, labels = batch
                emg_proc = preprocess_emg_enhanced(emg, normalizer, is_training=False).to(device)
                labels = labels.to(device).argmax(dim=1)
                outputs = model(emg_proc)
                val_loss += criterion(outputs, labels).item()

        avg_val_loss = val_loss / max(len(val_loader), 1)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}/{num_epochs} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Evaluation with TTA
    model.eval()
    test_preds = []
    test_true = []

    with torch.no_grad():
        if USE_TTA:
            model_tta = TTAWrapper(model, n_augments=8)
            for batch in tqdm(test_loader, desc="Testing [EMG+TTA]"):
                _, _, _, emg, labels = batch
                emg_proc = preprocess_emg_enhanced(emg, normalizer, is_training=False).to(device)
                labels = labels.to(device).argmax(dim=1)

                logits = model_tta(emg_proc)
                outputs = logits.exp()

                _, predicted = outputs.max(1)
                test_preds.extend(predicted.cpu().numpy())
                test_true.extend(labels.cpu().numpy())
        else:
            for batch in tqdm(test_loader, desc="Testing [EMG]"):
                _, _, _, emg, labels = batch
                emg_proc = preprocess_emg_enhanced(emg, normalizer, is_training=False).to(device)
                labels = labels.to(device).argmax(dim=1)
                outputs = model(emg_proc)

                _, predicted = outputs.max(1)
                test_preds.extend(predicted.cpu().numpy())
                test_true.extend(labels.cpu().numpy())

    acc = accuracy_score(test_true, test_preds)
    conf = confusion_matrix(test_true, test_preds, labels=list(range(num_classes)))
    precision, recall, f1_per_class, _ = precision_recall_fscore_support(
        test_true, test_preds, labels=list(range(num_classes)), average=None, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        test_true, test_preds, average="macro", zero_division=0
    )

    row = {
        "fold": i + 1,
        "test_subject": test_subject,
        "model": "Conformer_PathB",
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "precision_per_class": np.array_str(precision),
        "recall_per_class": np.array_str(recall),
        "f1_per_class": np.array_str(f1_per_class),
        "confusion_matrix": conf.tolist(),
    }

    rows_emg.append(row)
    pd.DataFrame(rows_emg).to_csv(csv_emg, index=False)

    print(f"✅ Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f}")
    print(f"📄 Results: {csv_emg}")

# Final summary
print(f"\n{'='*80}")
print("TRAINING COMPLETE - EMG PATH B (Conformer + Improvements)")
print(f"{'='*80}\n")

results_df = pd.read_csv(csv_emg)
mean_acc = results_df['accuracy'].mean()
std_acc = results_df['accuracy'].std()
mean_f1 = results_df['macro_f1'].mean()

print(f"Mean Accuracy:        {mean_acc:.4f} ± {std_acc:.4f}")
print(f"Mean Macro F1:        {mean_f1:.4f}")
print(f"Accuracy Range:       {results_df['accuracy'].min():.4f} - {results_df['accuracy'].max():.4f}")

print(f"\nComparison to baseline LSTM-MSA (0.5115):")
gain = mean_acc - 0.5115
pct = (gain / 0.5115 * 100)
print(f"  Absolute gain:      +{gain:.4f} ({pct:+.1f}%)")

print(f"\n📊 Results saved: {csv_emg}")
print(f"✨ Path B improvements applied successfully!")
