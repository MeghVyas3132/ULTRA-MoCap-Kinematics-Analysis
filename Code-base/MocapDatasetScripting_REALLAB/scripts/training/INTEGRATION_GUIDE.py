"""
INTEGRATION GUIDE: EMG Improvements into LOSO Training Loop

This document shows step-by-step how to integrate the new improvements from
emg_improvements.py into the existing conv1d_bigru_loso.py training script.

WORKFLOW:
1. Replace preprocessing functions (subject-wise normalization)
2. Add augmentation during data loading
3. Replace model architectures (Conformer, Dual-Branch)
4. Update training loop (AdamW, OneCycleLR, gradient clipping, combined loss)
5. Add validation logging per improvement
6. Evaluate performance gains iteratively

KEY INTEGRATION POINTS:
"""

import sys
import os

# Add emg_improvements module
sys.path.insert(0, '/Users/meghvyas/Desktop/research-paper/Code-base/MocapDatasetScripting_REALLAB/scripts/training')

from emg_improvements import (
    SubjectNormalizer,
    preprocess_emg_enhanced,
    emg_combined_features,
    EMGAugmenter,
    EMGConformer,
    DualBranchEMG,
    ContrastiveEmbeddingModel,
    SupervisedContrastiveLoss,
    TTAWrapper,
    create_loss_with_options,
    create_optimizer_and_scheduler,
    combined_loss,
    ChannelAttention,
)

# ============================================================
# STEP 1: PREPROCESSING ENHANCEMENT
# ============================================================
"""
BEFORE (existing code at line 451-489):
    def preprocess_emg_tensor(emg, fs=100, low=5, high=45, kernel_size=5):
        # Per-window normalization (data leakage risk)

AFTER (new approach):
    Use SubjectNormalizer for train/val/test split:
"""

def enhanced_emg_preprocessing(emg_batch, fs=100, normalizer=None, is_training=False):
    """
    Replacement for preprocess_emg_tensor (line 451).

    Usage in training loop:
        # Before epoch loop:
        normalizer = SubjectNormalizer()

        # First training batch: fit normalizer
        emg_proc = enhanced_emg_preprocessing(emg_batch, normalizer=normalizer, is_training=True)

        # Subsequent training batches: apply fitted stats
        emg_proc = enhanced_emg_preprocessing(emg_batch, normalizer=normalizer, is_training=False)

        # Validation/test: always use fitted stats
        emg_proc = enhanced_emg_preprocessing(emg_batch, normalizer=normalizer, is_training=False)
    """
    return preprocess_emg_enhanced(emg_batch, fs=fs, subject_normalizer=normalizer, is_training=is_training)


# ============================================================
# STEP 2: AUGMENTATION IN DATA LOADING
# ============================================================
"""
ADD to existing ImuJointPairDataset class (around line 331):
"""

class ImuJointPairDatasetWithAugmentation:
    """
    Extends existing ImuJointPairDataset to support augmentation.

    Usage:
        dataset = ImuJointPairDatasetWithAugmentation(
            config, subjects, window_length, window_overlap,
            split='train',
            emg_augment=True,
            augment_prob=0.7
        )
    """
    def __init__(self, base_dataset, emg_augment=False, augment_prob=0.7):
        self.base_dataset = base_dataset
        self.augmenter = EMGAugmenter() if emg_augment else None
        self.augment_prob = augment_prob

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        imu_acc, imu_gyr, joint_data, emg_data, label = self.base_dataset[idx]

        # Apply augmentation only during training
        if self.augmenter is not None and hasattr(self, 'is_training') and self.is_training:
            if np.random.rand() < self.augment_prob:
                emg_np = emg_data.numpy()
                emg_aug = self.augmenter(emg_np)
                emg_data = torch.tensor(emg_aug, dtype=emg_data.dtype)

        return imu_acc, imu_gyr, joint_data, emg_data, label


# ============================================================
# STEP 3: MODEL REPLACEMENT - OPTION A: CONFORMER
# ============================================================
"""
REPLACE existing EMGLSTMMSAModel (line 598) with:
"""

def build_emg_model_conformer(num_emg_channels, num_classes, EMG_D_MODEL=128, EMG_DROPOUT=0.2):
    """
    Replace EMG_MODEL_VARIANT == 'lstm_msa' with Conformer.

    How to integrate into existing code (around line 971-978):

        if EMG_MODEL_VARIANT == "conformer":
            model = build_emg_model_conformer(
                num_emg_channels, num_classes,
                EMG_D_MODEL=EMG_D_MODEL,
                EMG_DROPOUT=EMG_DROPOUT
            ).to(device)
        elif EMG_MODEL_VARIANT == "lstm_msa":
            # existing code
    """
    return EMGConformer(
        n_channels=num_emg_channels,
        n_classes=num_classes,
        d_model=EMG_D_MODEL,
        n_blocks=4,
        n_heads=4,
        dropout=EMG_DROPOUT,
    )


# ============================================================
# STEP 4: MODEL REPLACEMENT - OPTION B: DUAL-BRANCH
# ============================================================
"""
For Dual-Branch with handcrafted features + raw temporal:
"""

def build_emg_model_dual_branch(num_emg_channels, num_classes, EMG_D_MODEL=128, EMG_DROPOUT=0.2):
    """
    Dual-branch: raw LSTM/Conformer + handcrafted MLP.

    Usage in existing code:

        if EMG_MODEL_VARIANT == "dual_branch":
            model = build_emg_model_dual_branch(
                num_emg_channels, num_classes,
                EMG_D_MODEL=EMG_D_MODEL,
                EMG_DROPOUT=EMG_DROPOUT
            ).to(device)
    """
    return DualBranchEMG(
        n_channels=num_emg_channels,
        n_classes=num_classes,
        d_model=EMG_D_MODEL,
        n_handcraft_features=9 * num_emg_channels,  # 9 features per channel
        use_conformer=True,
        dropout=EMG_DROPOUT,
    )


# ============================================================
# STEP 5: TRAINING LOOP MODIFICATIONS
# ============================================================
"""
REPLACE existing training optimizer/scheduler (line 1014-1034) with:
"""

def setup_emg_training(
    model,
    num_train_batches,
    num_epochs,
    use_class_weights=True,
    class_weights=None,
    device='cpu',
    EMG_LR=1e-3,
    EMG_WEIGHT_DECAY=0.01,
):
    """
    Enhanced optimizer + scheduler setup for EMG training.

    Integration (replace line 1014-1034):

        if modality == "emg":
            optimizer, scheduler = setup_emg_training(
                model,
                len(train_loader),
                num_epochs,
                use_class_weights=EMG_USE_CLASS_WEIGHTS,
                class_weights=class_weights,
                device=device,
                EMG_LR=EMG_LR,
                EMG_WEIGHT_DECAY=EMG_WEIGHT_DECAY,
            )
            criterion = create_loss_with_options(
                num_classes,
                use_class_weights=use_class_weights,
                class_weights=class_weights,
                label_smoothing=0.1,
                device=device
            )
    """
    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        num_train_batches=num_train_batches,
        num_epochs=num_epochs,
        lr=EMG_LR,
        weight_decay=EMG_WEIGHT_DECAY,
        use_onecycle=True,  # OneCycleLR instead of ReduceLROnPlateau
    )

    criterion = create_loss_with_options(
        num_classes=model.classifier[-1].out_features if hasattr(model, 'classifier') else 5,
        use_class_weights=use_class_weights,
        class_weights=class_weights,
        label_smoothing=0.1,
        device=device
    )

    return optimizer, scheduler, criterion


# ============================================================
# STEP 6: TRAINING LOOP - EMG ONLY (Conformer Example)
# ============================================================
"""
New training loop for EMG with all improvements.
REPLACE the existing training loop (line 1040-1127) with this:
"""

def train_emg_improved(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    num_epochs=30,
    patience=6,
    model_path="model.pt",
    emg_normalizer=None,
    use_augmentation=False,
    augmenter=None,
    use_mixup=False,
):
    """
    Enhanced training loop for EMG with:
      - Subject-wise normalization
      - AdamW + OneCycleLR
      - Gradient clipping
      - Label smoothing + class weights
      - Optional augmentation (noise, scale, warp)
      - Optional mixup
    """
    best_val_loss = float('inf')
    patience_counter = 0

    normalizer_fitted = False

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            _, _, _, emg, labels = batch

            # Enhanced preprocessing with subject-wise normalization
            emg_proc = preprocess_emg_enhanced(
                emg,
                subject_normalizer=emg_normalizer,
                is_training=(not normalizer_fitted)  # fit on first batch
            )
            normalizer_fitted = True
            emg_proc = emg_proc.to(device)
            labels = labels.to(device).argmax(dim=1)

            # Optional augmentation
            if use_augmentation and augmenter is not None:
                emg_np = emg.numpy()
                emg_aug = augmenter(emg_np)
                emg_proc = torch.tensor(emg_aug, dtype=torch.float32).to(device)

            # Optional mixup
            if use_mixup:
                from emg_improvements import mixup_batch, mixup_criterion
                emg_mix, labels_a, labels_b, lam = mixup_batch(emg_proc, labels, alpha=0.2)
                outputs = model(emg_mix)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                outputs = model(emg_proc)
                loss = criterion(outputs, labels)

            # Backward pass with gradient clipping
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # OneCycleLR: step after each batch (not epoch)
            if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / max(len(train_loader), 1)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                _, _, _, emg, labels = batch
                emg_proc = preprocess_emg_enhanced(
                    emg,
                    subject_normalizer=emg_normalizer,
                    is_training=False
                ).to(device)
                labels = labels.to(device).argmax(dim=1)
                outputs = model(emg_proc)
                val_loss += criterion(outputs, labels).item()

        avg_val_loss = val_loss / max(len(val_loader), 1)

        # ReduceLROnPlateau strategy (if not using OneCycleLR)
        if not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print(f"  ✓ Best model saved (val_loss={avg_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return best_val_loss


# ============================================================
# STEP 7: CONTRASTIVE PRE-TRAINING PHASE
# ============================================================
"""
Optional: Pre-train with contrastive loss before supervised learning.
RUN THIS BEFORE the main training loop for ~10-20 epochs.
"""

def pretrain_with_contrastive_loss(
    train_loader,
    val_loader,
    device,
    num_epochs=20,
    embedding_dim=128,
    num_emg_channels=3,
    temperature=0.07,
):
    """
    Contrastive pre-training phase.

    Integration (NEW PHASE before main LOSO loop):

        print("\\n=== PHASE 1: Contrastive Pre-training ===")
        emb_model, pretrain_weights = pretrain_with_contrastive_loss(
            train_loader, val_loader, device,
            num_epochs=15,
            embedding_dim=128,
            num_emg_channels=num_emg_channels,
        )

        # Then load pretrained weights into main model (Phase 2)
        print("\\n=== PHASE 2: Supervised Fine-tuning ===")
        model = build_emg_model_conformer(...)
        model.load_state_dict(pretrain_weights, strict=False)
        # Optionally freeze backbone, train classifier only for 5 epochs
    """
    emb_model = ContrastiveEmbeddingModel(
        num_emg_channels,
        embedding_dim=embedding_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(emb_model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = SupervisedContrastiveLoss(temperature=temperature)

    for epoch in range(num_epochs):
        emb_model.train()
        running_loss = 0.0

        for batch in train_loader:
            _, _, _, emg, labels = batch
            emg_proc = preprocess_emg_enhanced(emg).to(device)
            labels = labels.to(device).argmax(dim=1)

            embeddings = emb_model(emg_proc)  # [B, embedding_dim], normalized
            loss = criterion(embeddings, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(emb_model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(len(train_loader), 1)
        print(f"[Contrastive] Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")

    return emb_model, emb_model.state_dict()


# ============================================================
# STEP 8: TEST-TIME AUGMENTATION (TTA)
# ============================================================
"""
Apply TTA during evaluation (inference) for +1-3% accuracy.
NO retraining needed.

Replace evaluation function (line 727-815) with:
"""

def evaluate_model_with_tta(
    model,
    test_loader,
    criterion,
    device,
    modality="emg",
    use_tta=False,
    n_augments=8,
):
    """
    Evaluation with optional TTA.

    Integration (replace line 1134-1142):

        metrics = evaluate_model_with_tta(
            model, test_loader, criterion, device,
            modality=modality,
            use_tta=(modality == "emg"),  # enable TTA for EMG only
            n_augments=8
        )
    """
    if use_tta:
        model_tta = TTAWrapper(model, n_augments=n_augments)
        model = model_tta
        test_model = model_tta
    else:
        test_model = model

    test_model.eval()
    test_preds, test_true = [], []
    test_loss = 0.0

    with torch.no_grad():
        for batch in test_loader:
            if modality == "emg":
                _, _, _, emg, labels = batch
                emg_proc = preprocess_emg_enhanced(emg).to(device)
                labels = labels.to(device).argmax(dim=1)

                if use_tta:
                    logits = test_model(emg_proc)
                    outputs = logits.exp()  # convert from log-probs
                else:
                    outputs = model(emg_proc)

            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = outputs.max(1)
            test_preds.extend(predicted.cpu().numpy())
            test_true.extend(labels.cpu().numpy())

    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

    acc = accuracy_score(test_true, test_preds)
    conf = confusion_matrix(test_true, test_preds)
    precision, recall, f1_per_class, _ = precision_recall_fscore_support(
        test_true, test_preds, average=None, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        test_true, test_preds, average='macro', zero_division=0
    )

    return {
        'accuracy': acc,
        'confusion_matrix': conf,
        'macro_precision': macro_p,
        'macro_recall': macro_r,
        'macro_f1': macro_f1,
        'test_loss': test_loss / max(len(test_loader), 1),
    }


# ============================================================
# STEP 9: FEATURE EXTRACTION FOR DUAL-BRANCH
# ============================================================
"""
If using Dual-Branch model, extract handcrafted features per window.
"""

class DataLoaderWithHandcraftedFeatures:
    """
    Wraps DataLoader to inject handcrafted EMG features per batch.
    """
    def __init__(self, base_loader, compute_features=True):
        self.base_loader = base_loader
        self.compute_features = compute_features

    def __iter__(self):
        for imu_acc, imu_gyr, joint_data, emg_data, labels in self.base_loader:
            if self.compute_features:
                # Extract handcrafted features for each EMG window in batch
                emg_feat_list = []
                for i in range(emg_data.shape[0]):
                    window = emg_data[i].numpy()  # [T, C]
                    feats = emg_combined_features(window)  # [9*C]
                    emg_feat_list.append(feats)

                emg_feat = torch.tensor(np.stack(emg_feat_list), dtype=torch.float32)
                yield imu_acc, imu_gyr, joint_data, emg_data, emg_feat, labels
            else:
                yield imu_acc, imu_gyr, joint_data, emg_data, None, labels

    def __len__(self):
        return len(self.base_loader)


# ============================================================
# COMPLETE INTEGRATION EXAMPLE
# ============================================================
"""
HOW TO INTEGRATE ALL IMPROVEMENTS INTO EXISTING SCRIPT:

1. Add import at top:
   from emg_improvements import *

2. Before LOSO loop (around line 920):
   # Initialize normalizer per fold
   emg_normalizer = SubjectNormalizer()
   augmenter = EMGAugmenter(prob_noise=0.5, prob_scale=0.5, prob_warp=0.3)

3. Model creation (around line 970):
   if modality == "emg":
       if EMG_MODEL_VARIANT == "conformer":
           model = EMGConformer(
               num_emg_channels, num_classes,
               d_model=EMG_D_MODEL, n_blocks=4,
               dropout=EMG_DROPOUT
           ).to(device)
       elif EMG_MODEL_VARIANT == "dual_branch":
           model = DualBranchEMG(
               num_emg_channels, num_classes,
               d_model=EMG_D_MODEL,
               n_handcraft_features=9*num_emg_channels,
               dropout=EMG_DROPOUT
           ).to(device)

4. Optimizer + Scheduler (around line 1014):
   if modality == "emg":
       optimizer, scheduler = create_optimizer_and_scheduler(
           model,
           len(train_loader), num_epochs,
           lr=EMG_LR, weight_decay=EMG_WEIGHT_DECAY,
           use_onecycle=True
       )
       criterion = create_loss_with_options(
           num_classes,
           use_class_weights=EMG_USE_CLASS_WEIGHTS,
           class_weights=class_weights if EMG_USE_CLASS_WEIGHTS else None,
           label_smoothing=0.1, device=device
       )

5. Training loop (around line 1040):
   # Use enhanced training function
   best_val_loss = train_emg_improved(
       model, train_loader, val_loader,
       optimizer, scheduler, criterion, device,
       num_epochs=num_epochs, patience=patience,
       model_path=model_path,
       emg_normalizer=emg_normalizer,
       use_augmentation=True,
       augmenter=augmenter,
       use_mixup=False  # optional
   )

6. Evaluation (around line 1134):
   metrics = evaluate_model_with_tta(
       model, test_loader, criterion, device,
       modality="emg",
       use_tta=True,  # enable TTA for EMG
       n_augments=8
   )
"""
