"""
EMG-Only Classification Strategic Improvements Module

This module implements all 10 priority improvements for EMG-only gesture classification:
  1. Subject-wise normalization (enhanced preprocessing)
  2. Frequency-domain EMG features
  3. AdamW + OneCycleLR scheduling
  4. EMG augmentation (noise, scale, warp)
  5. Conformer architecture (SOTA for biosignals)
  6. Dual-branch architecture (raw + handcrafted features)
  7. Supervised contrastive pre-training
  8. CWT scalogram branch
  9. Test-time augmentation (TTA)
  10. Channel attention + label smoothing

Ready for integration into existing LOSO training loop.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
from scipy.interpolate import interp1d
import pywt


# ============================================================
# 1. ENHANCED PREPROCESSING: Subject-Wise Normalization
# ============================================================

class SubjectNormalizer:
    """
    Manages subject-specific normalization statistics.
    Compute stats from training data, apply to train/val/test.
    """
    def __init__(self):
        self.train_mean = None
        self.train_std = None

    def fit(self, X_train):
        """
        X_train: [N, T, C] — compute per-channel statistics across all samples
        """
        # Mean and std across samples and time, per channel
        self.train_mean = X_train.mean(axis=(0, 1), keepdims=True)  # (1, 1, C)
        self.train_std = X_train.std(axis=(0, 1), keepdims=True) + 1e-6

    def transform(self, X):
        """Apply training statistics to X (train/val/test)."""
        assert self.train_mean is not None, "Must call fit() first"
        return (X - self.train_mean) / self.train_std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def preprocess_emg_enhanced(
    emg_tensor,
    fs=100,
    low=5,
    high=45,
    kernel_size=5,
    subject_normalizer=None,
    is_training=False
):
    """
    Enhanced EMG preprocessing:
      - Bandpass filter (5-45 Hz)
      - Rectification + smoothing
      - Subject-wise Z-score normalization (not per-window)

    emg_tensor: [B, T, C]
    subject_normalizer: SubjectNormalizer instance
    is_training: if True and normalizer not yet fit, fit it; else apply
    """
    from scipy.signal import butter, filtfilt

    B, T, C = emg_tensor.shape
    emg_np = emg_tensor.detach().cpu().numpy().astype(np.float32)

    def butter_bandpass(lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low_n = lowcut / nyq
        high_n = highcut / nyq
        b, a = butter(order, [low_n, high_n], btype="band")
        return b, a

    b, a = butter_bandpass(low, high, fs)
    emg_filtered = np.zeros_like(emg_np, dtype=np.float32)

    for bi in range(B):
        for ci in range(C):
            x = emg_np[bi, :, ci]
            x = filtfilt(b, a, x)
            x = np.abs(x)  # rectify
            x = np.convolve(
                x,
                np.ones(kernel_size, dtype=np.float32) / kernel_size,
                mode="same",
            )
            emg_filtered[bi, :, ci] = x

    # Subject-wise normalization instead of per-window
    if subject_normalizer is not None:
        if is_training:
            emg_norm = subject_normalizer.fit_transform(emg_filtered)
        else:
            emg_norm = subject_normalizer.transform(emg_filtered)
    else:
        # Fallback to per-window if normalizer not provided
        mean = emg_filtered.mean(axis=1, keepdims=True)
        std = emg_filtered.std(axis=1, keepdims=True) + 1e-6
        emg_norm = (emg_filtered - mean) / std

    return torch.tensor(emg_norm, dtype=emg_tensor.dtype, device=emg_tensor.device)


# ============================================================
# 2. FREQUENCY-DOMAIN EMG FEATURES
# ============================================================

def emg_time_domain_features(window, fs=100):
    """
    Time-domain features: RMS, MAV, ZC, WL
    window: [T, C]
    Returns: [4*C]
    """
    features = []
    for c in range(window.shape[1]):
        sig = window[:, c]

        # RMS (Root Mean Square)
        rms = np.sqrt(np.mean(sig ** 2))

        # MAV (Mean Absolute Value)
        mav = np.mean(np.abs(sig))

        # ZC (Zero Crossing count)
        zc = np.sum(np.diff(np.sign(sig)) != 0)

        # WL (Waveform Length)
        wl = np.sum(np.abs(np.diff(sig)))

        features.extend([rms, mav, zc, wl])

    return np.array(features, dtype=np.float32)  # [4*C]


def emg_frequency_domain_features(window, fs=100, nperseg=None):
    """
    Frequency-domain features: Mean Freq, Median Freq, Band Power
    window: [T, C]
    Returns: [5*C]  (MNF, MDF, BP_low, BP_mid, BP_high per channel)
    """
    if nperseg is None:
        nperseg = min(256, len(window))

    features = []
    for c in range(window.shape[1]):
        sig = window[:, c]

        # Welch PSD
        freqs, psd = signal.welch(sig, fs=fs, nperseg=nperseg)
        psd_norm = psd / (np.sum(psd) + 1e-8)

        # Mean Frequency (first moment)
        mnf = np.sum(freqs * psd_norm)

        # Median Frequency (50% power split)
        cumsum = np.cumsum(psd_norm)
        mdf = freqs[np.searchsorted(cumsum, 0.5)]

        # Band Power: low (20-150 Hz), mid (150-350 Hz), high (350-500 Hz)
        def band_power(f_low, f_high):
            mask = (freqs >= f_low) & (freqs <= f_high)
            if np.sum(mask) == 0:
                return 0.0
            return np.trapezoid(psd_norm[mask], freqs[mask]) if hasattr(np, 'trapezoid') else np.trapz(psd_norm[mask], freqs[mask])

        bp_low = band_power(20, 150)
        bp_mid = band_power(150, 350)
        bp_high = band_power(350, 500)

        features.extend([mnf, mdf, bp_low, bp_mid, bp_high])

    return np.array(features, dtype=np.float32)  # [5*C]


def emg_combined_features(window, fs=100):
    """
    Combined time + frequency domain features.
    window: [T, C]
    Returns: [9*C] = [4*C time-domain + 5*C freq-domain]
    """
    td_feats = emg_time_domain_features(window, fs)
    fd_feats = emg_frequency_domain_features(window, fs)
    return np.concatenate([td_feats, fd_feats])  # [9*C]


# ============================================================
# 3. EMG DATA AUGMENTATION
# ============================================================

class EMGAugmenter:
    """Multi-strategy augmentation for raw EMG windows (training only)."""

    def __init__(self, prob_noise=0.5, prob_scale=0.5, prob_warp=0.3, prob_dropout=0.2):
        self.prob_noise = prob_noise
        self.prob_scale = prob_scale
        self.prob_warp = prob_warp
        self.prob_dropout = prob_dropout

    def __call__(self, x_np):
        """
        x_np: [T, C] or [B, T, C] numpy array
        Returns augmented array (same shape)
        """
        # Handle batch
        if x_np.ndim == 3:
            return np.stack([self._augment_single(x_np[i]) for i in range(x_np.shape[0])])
        else:
            return self._augment_single(x_np)

    def _augment_single(self, x):
        """x: [T, C]"""
        x = x.copy()

        # 1. Gaussian noise (electrode noise simulation)
        if np.random.rand() < self.prob_noise:
            noise_std = 0.05 * x.std(axis=0, keepdims=True)
            x = x + np.random.normal(0, noise_std, x.shape)

        # 2. Amplitude scaling (gain drift)
        if np.random.rand() < self.prob_scale:
            scale = np.random.uniform(0.8, 1.2)
            x = x * scale

        # 3. Time warping (contraction speed variability)
        if np.random.rand() < self.prob_warp:
            x = self._time_warp(x)

        # 4. Channel dropout (electrode detachment)
        if np.random.rand() < self.prob_dropout:
            ch = np.random.randint(x.shape[1])
            x[:, ch] = 0.0

        return x

    @staticmethod
    def _time_warp(x):
        """x: [T, C]"""
        T = x.shape[0]
        stretch = np.random.uniform(0.9, 1.1)
        T_new = max(1, int(T * stretch))
        
        x_warped = np.zeros_like(x)
        for c in range(x.shape[1]):
            # Resample to T_new length, then resample back to T
            t_orig = np.linspace(0, 1, T)
            t_stretched = np.linspace(0, 1, T_new)
            f = interp1d(t_orig, x[:, c], kind="linear", fill_value="extrapolate")
            x_stretched = f(t_stretched)
            # Resample back to original length
            f2 = interp1d(t_stretched, x_stretched, kind="linear", fill_value="extrapolate")
            x_warped[:, c] = f2(t_orig)

        return x_warped


# ============================================================
# 4. CONFORMER ARCHITECTURE (SOTA Hybrid CNN+Transformer)
# ============================================================

class ConvolutionalBlock(nn.Module):
    """
    Pointwise linear + depthwise convolution + Gated Linear Unit.
    From Conformer paper (Gulati et al., 2021).
    """
    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.pw1 = nn.Linear(d_model, d_model * 2)  # pointwise → 2x channels
        self.dw = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model  # depthwise = groups == in_channels
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.pw2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()

    def forward(self, x):
        """x: [B, T, D]"""
        residual = x
        x = self.norm(x)

        # Gated linear unit
        x = self.pw1(x)  # [B, T, 2D]
        x, gate = x.chunk(2, dim=-1)
        x = x * gate.sigmoid()

        # Depthwise convolution
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.dw(x)
        x = self.bn(x)
        x = self.activation(x)
        x = x.transpose(1, 2)  # [B, T, D]

        # Linear projection + residual
        x = self.pw2(x)
        x = residual + self.dropout(x)
        return x


class ConformerBlock(nn.Module):
    """Full Conformer block: FF → Attention → Convolution → FF"""
    def __init__(self, d_model=128, n_heads=4, ff_mult=4,
                 conv_kernel=31, dropout=0.1):
        super().__init__()
        self.ff1 = self._ff_module(d_model, ff_mult, dropout)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads,
            dropout=dropout, batch_first=True
        )
        self.conv = ConvolutionalBlock(d_model, conv_kernel, dropout)
        self.ff2 = self._ff_module(d_model, ff_mult, dropout)
        self.norm_attn = nn.LayerNorm(d_model)

    @staticmethod
    def _ff_module(d, mult, dropout):
        return nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d * mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d * mult, d),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """x: [B, T, D]"""
        # FF + residual
        x = x + 0.5 * self.ff1(x)

        # Multi-head attention + residual
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out

        # Convolution module
        x = x + self.conv(x)

        # FF + residual
        x = x + 0.5 * self.ff2(x)
        return x


class EMGConformer(nn.Module):
    """
    Conformer for EMG gesture classification.
    Input: [B, T, C_emg]
    Output: [B, num_classes]
    """
    def __init__(self, n_channels, n_classes, d_model=128,
                 n_blocks=4, n_heads=4, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv1d(n_channels, d_model, kernel_size=7, padding=3),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList([
            ConformerBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_blocks)
        ])
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

    def forward(self, x):
        """x: [B, T, C]"""
        # Project to d_model
        x = x.transpose(1, 2)  # [B, C, T]
        x = self.input_proj(x)  # [B, D, T]
        x = x.transpose(1, 2)  # [B, T, D]

        # Conformer blocks
        for block in self.blocks:
            x = block(x)

        # Global average pooling + classification
        x = x.mean(dim=1)  # [B, D]
        return self.classifier(x)


# ============================================================
# 5. DUAL-BRANCH ARCHITECTURE (Raw + Handcrafted Features)
# ============================================================

class ChannelAttention(nn.Module):
    """Spatial attention over EMG channels (learns which channels matter)."""
    def __init__(self, n_channels, reduction=2):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(n_channels, max(1, n_channels // reduction)),
            nn.ReLU(),
            nn.Linear(max(1, n_channels // reduction), n_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """x: [B, T, C]"""
        # Average over time → [B, C]
        weights = self.attn(x.mean(dim=1))
        return x * weights.unsqueeze(1)  # [B, T, C]


class DualBranchEMG(nn.Module):
    """
    Branch 1: Raw temporal (LSTM/Conformer path)
    Branch 2: Handcrafted features (MLP path)
    Fusion: Concatenate + classifier
    """
    def __init__(self, n_channels, n_classes, d_model=128,
                 n_handcraft_features=None, use_conformer=True, dropout=0.2):
        super().__init__()

        self.n_channels = n_channels
        self.use_conformer = use_conformer

        # Branch 1: Raw temporal
        if use_conformer:
            # Use conformer blocks directly without final classifier
            self.input_proj = nn.Sequential(
                nn.Conv1d(n_channels, d_model, kernel_size=7, padding=3),
                nn.BatchNorm1d(d_model),
                nn.SiLU(),
            )
            self.blocks = nn.ModuleList([
                ConformerBlock(d_model, n_heads=4, dropout=dropout)
                for _ in range(3)
            ])
            self.temporal_norm = nn.LayerNorm(d_model)
            temporal_out_dim = d_model
        else:
            self.temporal_lstm = nn.LSTM(
                n_channels, d_model, num_layers=2,
                batch_first=True, bidirectional=True, dropout=dropout
            )
            self.temporal_attn = ChannelAttention(d_model * 2)
            temporal_out_dim = d_model * 2

        # Branch 2: Handcrafted features MLP
        if n_handcraft_features is None:
            n_handcraft_features = 9 * n_channels  # 9 features per channel

        self.feat_mlp = nn.Sequential(
            nn.Linear(n_handcraft_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(temporal_out_dim + 64, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 1.5),
            nn.Linear(128, n_classes),
        )

    def forward(self, x_raw, x_feat):
        """
        x_raw: [B, T, C]
        x_feat: [B, F] handcrafted features
        """
        if self.use_conformer:
            # Process with conformer blocks
            x = x_raw.transpose(1, 2)  # [B, C, T]
            x = self.input_proj(x)  # [B, D, T]
            x = x.transpose(1, 2)  # [B, T, D]
            for block in self.blocks:
                x = block(x)
            x = self.temporal_norm(x)
            temporal_feat = x.mean(dim=1)  # [B, D]
        else:
            lstm_out, _ = self.temporal_lstm(x_raw)  # [B, T, 2D]
            lstm_out = self.temporal_attn(lstm_out)
            temporal_feat = lstm_out.mean(dim=1)  # [B, 2D]

        # Feature branch
        feat_out = self.feat_mlp(x_feat)  # [B, 64]

        # Fuse
        fused = torch.cat([temporal_feat, feat_out], dim=-1)
        return self.classifier(fused)


# ============================================================
# 6. SUPERVISED CONTRASTIVE LEARNING
# ============================================================

class SupervisedContrastiveLoss(nn.Module):
    """
    Learns subject-invariant feature representations via contrastive loss.
    See Chen et al. 2020 "Supervised Contrastive Learning".
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.tau = temperature

    def forward(self, features, labels):
        """
        features: [B, D] L2-normalized embeddings
        labels: [B] gesture class indices
        """
        B = features.size(0)

        # Cosine similarity matrix
        sim = torch.matmul(features, features.T) / self.tau  # [B, B]

        # Create masks
        labels_expanded = labels.unsqueeze(1)
        pos_mask = (labels_expanded == labels_expanded.T).float()
        pos_mask.fill_diagonal_(0)  # exclude self

        neg_mask = 1.0 - pos_mask
        neg_mask.fill_diagonal_(0)

        # Numerator: exp(sim) for positives
        # Denominator: sum of all exp(sim)
        exp_sim = torch.exp(sim)

        pos_sum = (exp_sim * pos_mask).sum(dim=1)
        total_sum = exp_sim.sum(dim=1)

        loss = -torch.log((pos_sum / (total_sum + 1e-8)) + 1e-8)

        # Average over samples with at least 1 positive
        n_pos = pos_mask.sum(dim=1)
        valid_mask = (n_pos > 0).float()

        return (loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)


class ContrastiveEmbeddingModel(nn.Module):
    """Encoder that produces normalized embeddings for contrastive learning."""
    def __init__(self, n_channels, embedding_dim=128, d_model=128):
        super().__init__()
        self.encoder = EMGConformer(n_channels, embedding_dim, d_model=d_model, n_blocks=3)
        self.embedding_dim = embedding_dim

    def forward(self, x):
        """x: [B, T, C]"""
        emb = self.encoder(x)  # [B, embedding_dim]
        return F.normalize(emb, dim=-1)


# ============================================================
# 7. CONTINUOUS WAVELET TRANSFORM (CWT) SCALOGRAM BRANCH
# ============================================================

def compute_cwt_scalogram(window, wavelet='morl', num_scales=32, fs=100):
    """
    Compute CWT scalogram for one window.
    window: [T, C]
    Returns: [C, num_scales, T] scalogram
    """
    scales = np.geomspace(1, 128, num=num_scales)
    scalograms = []

    for c in range(window.shape[1]):
        sig = window[:, c]
        coeffs, _ = pywt.cwt(sig, scales, wavelet, sampling_period=1/fs)
        scalograms.append(np.abs(coeffs))  # [num_scales, T]

    return np.stack(scalograms, axis=0)  # [C, num_scales, T]


class CWTBranch(nn.Module):
    """Lightweight CNN for CWT scalogram features."""
    def __init__(self, n_channels, n_scales, out_dim=64, dropout=0.1):
        super().__init__()
        self.cnn = nn.Sequential(
            # Input: [B, n_channels, n_scales, T]
            nn.Conv2d(n_channels, 32, kernel_size=(3, 7), padding=(1, 3)),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d((2, 4)),  # downsample

            nn.Conv2d(32, 64, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d((2, 2)),

            nn.AdaptiveAvgPool2d((4, 1)),
        )

        self.proj = nn.Linear(64 * 4, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: [B, C, num_scales, T]"""
        out = self.cnn(x)  # [B, 64, 4, 1]
        out = out.flatten(1)  # [B, 256]
        return self.proj(self.dropout(out))  # [B, out_dim]


# ============================================================
# 8. TEST-TIME AUGMENTATION (TTA)
# ============================================================

class TTAWrapper(nn.Module):
    """Ensemble predictions from augmented test views (inference only)."""
    def __init__(self, model, n_augments=8, augment_fn=None):
        super().__init__()
        self.model = model
        self.n_aug = n_augments
        self.augment_fn = augment_fn or self._default_augment

    def _default_augment(self, x):
        """Mild inference-safe augmentation: small noise + scale."""
        noise = torch.randn_like(x) * 0.02 * x.std()
        scale = torch.empty(x.size(0), 1, x.size(2)).uniform_(0.95, 1.05).to(x.device)
        return x * scale + noise

    @torch.no_grad()
    def forward(self, x):
        """
        x: [B, T, C]
        Returns ensemble prediction (log-probs)
        """
        logits = [self.model(x)]

        for _ in range(self.n_aug - 1):
            x_aug = self.augment_fn(x)
            logits.append(self.model(x_aug))

        # Average softmax probabilities
        probs = torch.stack([F.softmax(l, dim=-1) for l in logits], dim=0)
        ensemble_probs = probs.mean(0)
        return torch.log(ensemble_probs + 1e-10)


# ============================================================
# 9. COMBINED TRAINING UTILITIES
# ============================================================

def create_loss_with_options(
    num_classes,
    use_class_weights=False,
    class_weights=None,
    label_smoothing=0.1,
    device='cpu'
):
    """Create cross-entropy loss with optional class weighting + label smoothing."""
    if use_class_weights and class_weights is not None:
        return nn.CrossEntropyLoss(
            weight=class_weights.to(device),
            label_smoothing=label_smoothing
        )
    else:
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def create_optimizer_and_scheduler(
    model,
    num_train_batches,
    num_epochs,
    lr=1e-3,
    weight_decay=1e-4,
    use_onecycle=True,
):
    """Create AdamW optimizer with optional OneCycleLR scheduling."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    if use_onecycle:
        from torch.optim.lr_scheduler import OneCycleLR
        scheduler = OneCycleLR(
            optimizer,
            max_lr=lr,
            steps_per_epoch=num_train_batches,
            epochs=num_epochs,
            pct_start=0.3,  # 30% warmup
            anneal_strategy='cos',
        )
    else:
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )

    return optimizer, scheduler


def combined_loss(logits, embeddings, labels,
                  alpha=0.7, temperature=0.07, label_smoothing=0.1):
    """
    Combined cross-entropy + supervised contrastive loss.
    For joint training (contrastive pre-training + CE fine-tuning).

    logits: [B, num_classes]
    embeddings: [B, D] normalized embeddings
    labels: [B]
    alpha: weight on CE loss
    """
    ce_loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)

    scl = SupervisedContrastiveLoss(temperature)
    contrastive_loss = scl(embeddings, labels)

    return alpha * ce_loss + (1.0 - alpha) * contrastive_loss


# ============================================================
# 10. BATCH-LEVEL AUGMENTATION UTILITIES
# ============================================================

def mixup_batch(x, y, alpha=0.2):
    """
    Mixup augmentation for time series.
    Returns: x_mix, y_a, y_b, lam
    Loss: lam * loss(pred, y_a) + (1-lam) * loss(pred, y_b)
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)

    x_mix = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return x_mix, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================
# USAGE EXAMPLES (documentation)
# ============================================================
"""
INTEGRATION EXAMPLES:

1. Enhanced EMG preprocessing with subject normalization:
   ---
   normalizer = SubjectNormalizer()
   X_train_norm = preprocess_emg_enhanced(X_train, subject_normalizer=normalizer, is_training=True)
   X_test_norm = preprocess_emg_enhanced(X_test, subject_normalizer=normalizer, is_training=False)

2. Add handcrafted features for dual-branch:
   ---
   for batch in train_loader:
       _, _, _, emg, labels = batch
       emg_feat = np.array([emg_combined_features(e.numpy()) for e in emg])
       emg_feat_tensor = torch.tensor(emg_feat, dtype=torch.float32)

       emg_raw_proc = preprocess_emg_enhanced(emg, ...)
       outputs = model(emg_raw_proc, emg_feat_tensor)

3. Use Conformer instead of LSTM-MSA:
   ---
   model = EMGConformer(
       n_channels=3,
       n_classes=5,
       d_model=128,
       n_blocks=4,
       dropout=0.2
   )

4. Supervised contrastive pre-training:
   ---
   # Phase 1: Pre-train with contrastive loss
   emb_model = ContrastiveEmbeddingModel(n_channels=3, embedding_dim=128)
   optimizer = torch.optim.AdamW(emb_model.parameters(), lr=1e-3)
   criterion = SupervisedContrastiveLoss(temperature=0.07)

   for epoch in range(10):
       for batch in train_loader:
           embs = emb_model(emg_proc)
           loss = criterion(embs, labels)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()

   # Phase 2: Freeze backbone, train classifier
   model = EMGConformer(...)
   model.encoder.load_state_dict(emb_model.encoder.state_dict())
   for param in model.encoder.parameters():
       param.requires_grad = False

5. Test-time augmentation:
   ---
   model_tta = TTAWrapper(model, n_augments=8)

   @torch.no_grad()
   def predict_with_tta(x):
       return model_tta(x).softmax(-1)

6. Data augmentation during training:
   ---
   augmenter = EMGAugmenter(prob_noise=0.5, prob_scale=0.5, prob_warp=0.3)

   for batch in train_loader:
       _, _, _, emg, labels = batch
       emg_np = emg.numpy()
       emg_aug = augmenter(emg_np)
       emg_aug_tensor = torch.tensor(emg_aug, dtype=torch.float32)
       # train with augmented data

7. Combined loss for joint training:
   ---
   loss = combined_loss(
       logits, embeddings, labels,
       alpha=0.7,  # 70% CE, 30% contrastive
       temperature=0.07
   )
"""
