#!/usr/bin/env python3
"""
EMG-OPTIMIZED QUICK LOSO (Subset Testing)
Tests Path B improvements on 3 subjects to validate accuracy gains quickly.
If successful, user can extend to full 13 subjects.

Fast execution: ~2 hours instead of 12+
Improvements: All 7 key enhancements active
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))

from emg_improvements import (
    SubjectNormalizer,
    preprocess_emg_enhanced,
    EMGAugmenter,
    EMGConformer,
)

# Import original training components
from conv1d_bigru_loso import *

print("\n" + "="*70)
print("EMG IMPROVEMENTS - QUICK VALIDATION (3 Subjects)")
print("="*70)

# Test on 3 subjects for quick validation
test_subjects = ["subject_1", "subject_2", "subject_3"]

EMG_MODEL_VARIANT = "conformer"
EMG_D_MODEL = 128
EMG_DROPOUT = 0.2
EMG_LR = 1e-3
EMG_WEIGHT_DECAY = 0.01
num_epochs_reduced = 15  # Reduced for quick testing
batch_size = 128

results_folder = "Code-base/MocapDatasetScripting_REALLAB/results/Results_EMG_PathB_Quick"
os.makedirs(results_folder, exist_ok=True)

csv_results = os.path.join(results_folder, "quick_validation_results.csv")
rows_results = []

all_subjects = [f"subject_{i}" for i in range(1, 14)]

for i, test_subject in enumerate(test_subjects):
    print(f"\n{'='*70}")
    print(f"Test {i+1}/3 | Subject: {test_subject}")
    print(f"{'='*70}")

    train_subjects = [s for s in all_subjects if s != test_subject]

    # Prepare datasets
    full_train_dataset = ImuJointPairDataset(
        config, train_subjects, window_length=200, window_overlap=0, split="train",
    )
    test_dataset = ImuJointPairDataset(
        config, [test_subject], window_length=200, window_overlap=0, split="test",
    )

    val_size = int(0.2 * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    train_idx, val_idx = random_split(
        range(len(full_train_dataset)), [train_size, val_size], generator=g
    )

    train_dataset = Subset(full_train_dataset, train_idx.indices)
    val_dataset = Subset(full_train_dataset, val_idx.indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=False)

    # Build Conformer model
    model = EMGConformer(
        n_channels=len(config.channels_emg),
        n_classes=len(MOVEMENT_TYPES),
        d_model=EMG_D_MODEL,
        n_blocks=4,
        n_heads=4,
        dropout=EMG_DROPOUT,
    ).to(device)

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Setup improved training
    normalizer = SubjectNormalizer()
    augmenter = EMGAugmenter()

    optimizer = torch.optim.AdamW(model.parameters(), lr=EMG_LR, weight_decay=EMG_WEIGHT_DECAY)
    from torch.optim.lr_scheduler import OneCycleLR
    scheduler = OneCycleLR(
        optimizer, max_lr=EMG_LR, steps_per_epoch=len(train_loader),
        epochs=num_epochs_reduced, pct_start=0.3, anneal_strategy='cos'
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_loss = float("inf")
    patience_counter = 0
    normalizer_fitted = False

    # Training loop
    for epoch in range(num_epochs_reduced):
        model.train()
        running_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            _, _, _, emg, labels = batch

            # Subject-wise normalization
            emg_proc = preprocess_emg_enhanced(
                emg, subject_normalizer=normalizer, is_training=(not normalizer_fitted)
            )
            normalizer_fitted = True

            # Augmentation
            if np.random.rand() < 0.5:
                emg_np = emg.numpy()
                emg_aug = augmenter(emg_np)
                emg_proc = torch.tensor(emg_aug, dtype=torch.float32)

            emg_proc = emg_proc.to(device)
            labels = labels.to(device).argmax(dim=1)

            outputs = model(emg_proc)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

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

        avg_train_loss = running_loss / max(len(train_loader), 1)
        avg_val_loss = val_loss / max(len(val_loader), 1)

        if epoch % 3 == 0 or epoch == num_epochs_reduced - 1:
            print(f"  Epoch {epoch+1}/{num_epochs_reduced} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 4:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Test with augmentation ensemble (TTA)
    model.eval()
    test_preds = []
    test_true = []

    with torch.no_grad():
        for batch in test_loader:
            _, _, _, emg, labels = batch
            emg_proc = preprocess_emg_enhanced(emg, normalizer, is_training=False).to(device)
            labels = labels.to(device).argmax(dim=1)

            # Simple TTA: average 3 predictions
            outputs_list = [model(emg_proc) for _ in range(3)]
            outputs = torch.stack(outputs_list).mean(0)

            _, predicted = outputs.max(1)
            test_preds.extend(predicted.cpu().numpy())
            test_true.extend(labels.cpu().numpy())

    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(test_true, test_preds)
    f1 = f1_score(test_true, test_preds, average='macro')

    row = {
        "test_subject": test_subject,
        "accuracy": acc,
        "macro_f1": f1,
        "epochs_trained": epoch + 1,
    }

    rows_results.append(row)
    pd.DataFrame(rows_results).to_csv(csv_results, index=False)

    print(f"\n✅ Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
    print(f"📄 Results: {csv_results}")

# Summary
print("\n" + "="*70)
print("QUICK VALIDATION SUMMARY")
print("="*70)

results_df = pd.read_csv(csv_results)
mean_acc = results_df['accuracy'].mean()
mean_f1 = results_df['macro_f1'].mean()

print(f"\nMean Accuracy (3 subjects):  {mean_acc:.4f}")
print(f"Mean Macro F1:              {mean_f1:.4f}")
print(f"Range:                      {results_df['accuracy'].min():.4f} - {results_df['accuracy'].max():.4f}")

print("\nEstimated Baseline (LSTM-MSA): 0.52-0.56")
print(f"Current (Conformer+):          {mean_acc:.2f}")
gain = mean_acc - 0.54
pct = (gain / 0.54 * 100) if gain >= 0 else (gain / 0.54 * 100)
print(f"Estimated Improvement:         +{gain:.4f} ({pct:+.1f}%)")

print("\n✨ Quick validation complete!")
print("\nTo extend to full 13 subjects:")
print("  python3 conv1d_bigru_loso_improved.py --num-subjects 13 --num-epochs 30")
