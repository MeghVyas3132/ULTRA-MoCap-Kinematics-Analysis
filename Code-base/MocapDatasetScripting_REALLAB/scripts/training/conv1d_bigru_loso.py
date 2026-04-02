"""
ULTRA-MoCap Upper Extremity Classification
Conv1D + BiGRU Encoder with IMU-only, EMG-only, and IMU+EMG Early Fusion

This script:
  - Shards HDF5 recordings into windowed CSVs
  - Builds a PyTorch Dataset over IMU, EMG, and joint data
  - Trains:
        (1) EMG-only Conv1D+BiGRU
        (2) IMU-only Conv1D+BiGRU
        (3) IMU+EMG early-fusion Conv1D+BiGRU
  - Runs Leave-One-Subject-Out (LOSO) cross-validation
  - Saves per-fold CSVs and a combined summary

Designed as a single-file, publication-ready reference implementation.
"""

# ============================================================
# 0. Colab Drive Mount (safe no-op outside Colab)
# ============================================================
try:
    from google.colab import drive  # type: ignore
    drive.mount('/content/drive')
except Exception:
    # Not running in Colab; skip mount.
    pass

# ============================================================
# 1. Imports
# ============================================================
import os
import re
import csv
import math
import random
import shutil
import ast
import argparse

import h5py
import numpy as np
import pandas as pd

from scipy.signal import butter, filtfilt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, random_split, Subset
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

# ============================================================
# 2. Reproducibility
# ============================================================
SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # deterministic cuBLAS (if supported)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def env_flag(name, default=False):
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


FAST_MODE = env_flag("FAST_MODE", default=False)
if FAST_MODE and torch.cuda.is_available():
    # Fast-mode favors throughput over strict determinism.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
else:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Torch generator for deterministic splits
g = torch.Generator().manual_seed(SEED)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda":
    matmul_precision = os.getenv("MATMUL_PRECISION", "high").strip().lower()
    if matmul_precision in {"high", "medium"}:
        torch.set_float32_matmul_precision(matmul_precision)
    print("FAST_MODE:", FAST_MODE)


def parse_cli_args():
    """
    Parse known CLI arguments while remaining notebook-friendly.
    """
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tag all outputs with _smoke and isolate partial test artifacts.",
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Optional override for number of LOSO folds to execute.",
    )
    parser.add_argument(
        "--smoke-subject",
        type=str,
        default=None,
        help="Optional single test subject for smoke execution.",
    )
    args, _ = parser.parse_known_args()
    return args


CLI_ARGS = parse_cli_args()


# ============================================================
# 3. Configuration
# ============================================================
class Config:
    """
    Simple config container.
    """

    def __init__(self, **kwargs):
        self.channels_imu_acc = kwargs.get("channels_imu_acc", [])
        self.channels_imu_gyr = kwargs.get("channels_imu_gyr", [])
        self.channels_joints = kwargs.get("channels_joints", [])
        self.channels_emg = kwargs.get("channels_emg", [])
        self.seed = kwargs.get("seed", 42)
        self.data_folder_name = kwargs.get("data_folder_name", "data.h5")
        self.dataset_root = kwargs.get("dataset_root", "./datasets")
        self.imu_transforms = kwargs.get("imu_transforms", [])
        self.joint_transforms = kwargs.get("joint_transforms", [])
        self.emg_transforms = kwargs.get("emg_transforms", [])
        self.input_format = kwargs.get("input_format", "csv")


# NOTE: Updated to local paths in this workspace
config = Config(
    data_folder_name="Dataset/ULTra-MoCap-processed/All_subjects_data.h5",
    dataset_root="Code-base/MocapDatasetScripting_REALLAB/datasets",
    input_format="csv",
    channels_imu_acc=[
        "ACCX1", "ACCY1", "ACCZ1",
        "ACCX2", "ACCY2", "ACCZ2",
        "ACCX3", "ACCY3", "ACCZ3",
        "ACCX4", "ACCY4", "ACCZ4",
        "ACCX5", "ACCY5", "ACCZ5",
        "ACCX6", "ACCY6", "ACCZ6",
    ],
    channels_imu_gyr=[
        "GYROX1", "GYROY1", "GYROZ1",
        "GYROX2", "GYROY2", "GYROZ2",
        "GYROX3", "GYROY3", "GYROZ3",
        "GYROX4", "GYROY4", "GYROZ4",
        "GYROX5", "GYROY5", "GYROZ5",
        "GYROX6", "GYROY6", "GYROZ6",
    ],
    channels_joints=["elbow_flex_r", "arm_flex_r", "arm_add_r"],
    channels_emg=["IM EMG4", "IM EMG5", "IM EMG6"],
)

# Optional environment overrides for Colab / remote runs.
if os.getenv("DATA_H5_PATH"):
    config.data_folder_name = os.getenv("DATA_H5_PATH", config.data_folder_name)
if os.getenv("DATASET_ROOT"):
    config.dataset_root = os.getenv("DATASET_ROOT", config.dataset_root)
print("Data H5:", config.data_folder_name)
print("Dataset root:", config.dataset_root)


# ============================================================
# 4. Sharding HDF5 to Windowed CSVs
# ============================================================
class DataSharder:
    """
    Reads subject data from a single HDF5 file and shards into
    windowed CSV segments for downstream training.
    """

    def __init__(self, config: Config, split: str):
        self.config = config
        self.h5_file_path = config.data_folder_name
        self.split = split
        self.window_length = None
        self.window_overlap = None

    def load_data(self, subjects, window_length, window_overlap, dataset_name):
        print(
            f"Processing subjects: {subjects} | "
            f"window_length={window_length}, overlap={window_overlap}"
        )

        self.window_length = window_length
        self.window_overlap = window_overlap
        self._process_and_save_patients_h5(subjects, dataset_name)

    def _process_and_save_patients_h5(self, subjects, dataset_name):
        with h5py.File(self.h5_file_path, "r") as h5_file:
            dataset_folder = os.path.join(
                self.config.dataset_root, dataset_name, self.split
            ).replace("subject", "").replace("__", "_")
            print("Dataset folder:", dataset_folder)

            if os.path.exists(dataset_folder):
                print("Dataset exists, skipping sharding...")
                return

            os.makedirs(dataset_folder, exist_ok=True)
            print("Created dataset folder:", dataset_folder)

            for subject_id in tqdm(subjects, desc="Processing subjects"):
                subject_key = subject_id
                if subject_key not in h5_file:
                    print(f"Subject {subject_key} not found in HDF5. Skipping.")
                    continue

                subject_data = h5_file[subject_key]
                session_keys = list(subject_data.keys())

                for session_id in session_keys:
                    session_data_group = subject_data[session_id]

                    for session_speed in session_data_group.keys():
                        session_data = session_data_group[session_speed]

                        imu_data, imu_columns = self._extract_channel_data(
                            session_data,
                            self.config.channels_imu_acc + self.config.channels_imu_gyr,
                        )
                        emg_data, emg_columns = self._extract_channel_data(
                            session_data, self.config.channels_emg
                        )
                        joint_data, joint_columns = self._extract_channel_data(
                            session_data, self.config.channels_joints
                        )

                        self._save_windowed_data(
                            imu_data=imu_data,
                            emg_data=emg_data,
                            joint_data=joint_data,
                            subject_key=subject_key,
                            session_id=session_id,
                            session_speed=session_speed,
                            dataset_folder=dataset_folder,
                            imu_columns=imu_columns,
                            emg_columns=emg_columns,
                            joint_columns=joint_columns,
                        )

    def _save_windowed_data(
        self,
        imu_data,
        emg_data,
        joint_data,
        subject_key,
        session_id,
        session_speed,
        dataset_folder,
        imu_columns,
        emg_columns,
        joint_columns,
    ):
        window_size = self.window_length
        overlap = self.window_overlap
        step_size = window_size - overlap

        csv_file_path = os.path.join(dataset_folder, "..", f"{self.split}_info.csv")
        os.makedirs(dataset_folder, exist_ok=True)

        csv_headers = ["file_name", "file_path"]
        file_exists = os.path.isfile(csv_file_path)

        with open(csv_file_path, mode="a", newline="") as csv_file:
            writer = csv.writer(csv_file)

            if not file_exists:
                writer.writerow(csv_headers)

            total_data_length = min(
                imu_data.shape[1], emg_data.shape[1], joint_data.shape[1]
            )

            # For longer recordings, skip first 2000 samples (warm-up)
            start = 2000 if total_data_length > 4000 else 0

            for i in range(start, total_data_length - window_size + 1, step_size):
                imu_window = imu_data[:, i : i + window_size]
                emg_window = emg_data[:, i : i + window_size]
                joint_window = joint_data[:, i : i + window_size]

                if (
                    imu_window.shape[1] != window_size
                    or emg_window.shape[1] != window_size
                    or joint_window.shape[1] != window_size
                ):
                    print(f"Skipping window {i} due to mismatched shapes.")
                    continue

                imu_df = pd.DataFrame(imu_window.T, columns=imu_columns)
                emg_df = pd.DataFrame(emg_window.T, columns=emg_columns)
                joint_df = pd.DataFrame(joint_window.T, columns=joint_columns)

                combined_df = pd.concat([imu_df, emg_df, joint_df], axis=1)

                file_name = (
                    f"{subject_key}_{session_id}_{session_speed}"
                    f"_win_{i}_ws{window_size}_ol{overlap}.csv"
                )
                file_path = os.path.join(dataset_folder, file_name)
                combined_df.to_csv(file_path, index=False)

                writer.writerow([file_name, file_path])

    def _extract_channel_data(self, session_data, channels):
        """
        Extracts per-channel data from a (possibly compound) HDF5 dataset
        and linearly interpolates NaNs.

        Returns:
          data: np.array [C, T]
          column_names: list[str] (channels that were actually found)
        """
        extracted_data = []
        new_column_names = []

        if isinstance(session_data, h5py.Dataset):
            # Case 1: Compound dataset (named fields)
            if session_data.dtype.names:
                column_names = list(session_data.dtype.names)
                for channel in channels:
                    if channel in column_names:
                        channel_data = session_data[channel][:]
                        channel_data = pd.to_numeric(channel_data, errors="coerce")
                        df = pd.DataFrame(channel_data)
                        df_interp = df.interpolate(
                            method="linear", axis=0, limit_direction="both"
                        )
                        extracted_data.append(df_interp.to_numpy().flatten())
                        new_column_names.append(channel)
                    else:
                        print(f"Channel {channel} not found in compound dataset.")
            else:
                # Case 2: Simple dataset with 'column_names' attribute
                column_names = session_data.attrs.get("column_names", [])
                column_names = list(column_names)
                assert len(column_names) > 0, "column_names not found in dataset attrs."

                for channel in channels:
                    if channel in column_names:
                        col_idx = column_names.index(channel)
                        new_column_names.append(channel)
                        channel_data = session_data[:, col_idx]
                        channel_data = pd.to_numeric(channel_data, errors="coerce")
                        df = pd.DataFrame(channel_data)
                        df_interp = df.interpolate(
                            method="linear", axis=0, limit_direction="both"
                        )
                        extracted_data.append(df_interp.to_numpy().flatten())
                    else:
                        print(f"Channel {channel} not found in session data.")

        return np.array(extracted_data), new_column_names


# ============================================================
# 5. Dataset: ImuJointPairDataset
# ============================================================
MOVEMENT_TYPES = ["OR", "EF", "ER", "CB", "AS"]
MOVEMENT_TYPE_MAP = {m: i for i, m in enumerate(MOVEMENT_TYPES)}


def compute_emg_features(window: np.ndarray) -> np.ndarray:
        """
        Compute handcrafted EMG features for one window.

        Args:
            window: np.ndarray with shape [T, C]

        Returns:
            np.ndarray with shape [4*C] containing per-channel:
            RMS, MAV, zero-crossings, and waveform length.
        """
        if window.ndim != 2:
                raise ValueError(f"Expected EMG window shape [T,C], got {window.shape}")

        rms = np.sqrt(np.mean(window ** 2, axis=0))
        mav = np.mean(np.abs(window), axis=0)
        zc = np.sum(np.diff(np.sign(window), axis=0) != 0, axis=0).astype(np.float32)
        wl = np.sum(np.abs(np.diff(window, axis=0)), axis=0)

        return np.concatenate([rms, mav, zc, wl], axis=0).astype(np.float32)


class ImuJointPairDataset(Dataset):
    """
    CSV-based dataset for IMU + joints + EMG windows.

    Expects sharded CSVs laid out as:
      <dataset_root>/<dataset_name>/<split>/*.csv
    plus a <split>_info.csv that logs file_name and file_path.
    """

    def __init__(
        self,
        config: Config,
        subjects,
        window_length,
        window_overlap,
        split="train",
        dataset_name="dataset",
        transforms=None,
    ):
        self.config = config
        self.split = split
        self.subjects = subjects
        self.window_length = window_length
        self.window_overlap = window_overlap if split == "train" else 0
        self.input_format = config.input_format
        self.channels_imu_acc = config.channels_imu_acc
        self.channels_imu_gyr = config.channels_imu_gyr
        self.channels_joints = config.channels_joints
        self.channels_emg = config.channels_emg
        self.transforms = (
            transforms
            if transforms is not None
            else {"imu": [], "joint": [], "emg": []}
        )
        self.dataset_name = dataset_name
        self.emg_feature_cache = {}

        subjects_str = "_".join(map(str, subjects)).replace("subject", "").replace(
            "__", "_"
        )
        self.dataset_folder_name = (
            f"{dataset_name}_wl{self.window_length}_ol{self.window_overlap}_"
            f"{self.split}{subjects_str}"
        )
        self.root_dir = os.path.join(self.config.dataset_root, self.dataset_folder_name)

        self._ensure_resharded(subjects)

        info_path = os.path.join(self.root_dir, f"{self.split}_info.csv")
        self.data = pd.read_csv(info_path)

    def _ensure_resharded(self, subjects):
        if not os.path.exists(self.root_dir):
            print(f"Sharded data not found at {self.root_dir}. Resharding...")
            sharder = DataSharder(self.config, self.split)
            sharder.load_data(
                subjects,
                window_length=self.window_length,
                window_overlap=self.window_overlap,
                dataset_name=self.dataset_folder_name,
            )
        else:
            print(f"Sharded data found at {self.root_dir}. Skipping resharding.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_name = self.data.iloc[idx, 0]
        file_path = os.path.join(self.root_dir, self.split, file_name)

        if self.input_format == "csv":
            combined_data = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported input format: {self.input_format}")

        imu_acc, imu_gyr, joint_data, emg_data, emg_features = self._extract_and_transform(
            combined_data,
            cache_key=file_name,
        )
        movement_label = self._extract_movement_label(file_name)

        return imu_acc, imu_gyr, joint_data, emg_data, emg_features, movement_label

    def _extract_and_transform(self, combined_data, cache_key=None):
        imu_acc = self._extract_channels(combined_data, self.channels_imu_acc)
        imu_gyr = self._extract_channels(combined_data, self.channels_imu_gyr)
        joint_data = self._extract_channels(combined_data, self.channels_joints)
        emg_data_np = self._extract_channels(combined_data, self.channels_emg)

        if cache_key is not None and cache_key in self.emg_feature_cache:
            emg_features = self.emg_feature_cache[cache_key]
        else:
            emg_features = torch.tensor(
                compute_emg_features(emg_data_np),
                dtype=torch.float32,
            )
            if cache_key is not None:
                self.emg_feature_cache[cache_key] = emg_features

        imu_acc = self._apply_transforms(imu_acc, self.transforms.get("imu", []))
        imu_gyr = self._apply_transforms(imu_gyr, self.transforms.get("imu", []))
        joint_data = self._apply_transforms(joint_data, self.transforms.get("joint", []))
        emg_data = self._apply_transforms(emg_data_np, self.transforms.get("emg", []))

        return imu_acc, imu_gyr, joint_data, emg_data, emg_features.clone()

    def _extract_movement_label(self, file_name: str) -> torch.Tensor:
        # NOTE: You confirmed this pattern is correct for your filenames.
        m = re.search(r"_(OR|EF|ER|CB|AS)_", file_name)
        if m:
            movement_type = m.group(1)
            idx = MOVEMENT_TYPE_MAP[movement_type]
            one_hot = torch.zeros(len(MOVEMENT_TYPES))
            one_hot[idx] = 1.0
            return one_hot
        raise ValueError(f"Unknown movement type in filename: {file_name}")

    @staticmethod
    def _extract_channels(combined_data, channels):
        return combined_data[channels].values

    @staticmethod
    def _apply_transforms(data, transforms):
        for t in transforms:
            data = t(data)
        return torch.tensor(data, dtype=torch.float32)


# ============================================================
# 6. Preprocessing: EMG & IMU
# ============================================================
def filter_emg_tensor(emg, fs=100, low=5, high=45, kernel_size=5):
    """
    sEMG preprocessing at 100 Hz:
      - 5–45 Hz bandpass
      - rectification
    - moving-average smoothing
    emg: [B, T, C]
    """
    B, T, C = emg.shape
    emg_np = emg.detach().cpu().numpy().astype(np.float32)

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
            x = np.abs(x)
            x = np.convolve(
                x,
                np.ones(kernel_size, dtype=np.float32) / kernel_size,
                mode="same",
            )
            emg_filtered[bi, :, ci] = x

    return torch.tensor(emg_filtered, dtype=emg.dtype, device=emg.device)


def preprocess_emg_tensor(
    emg,
    fs=100,
    low=5,
    high=45,
    kernel_size=5,
    train_mean=None,
    train_std=None,
):
    """
    EMG preprocessing + normalization.
    If train_mean/train_std are provided, apply train-only normalization stats.
    """
    emg_filtered = filter_emg_tensor(
        emg,
        fs=fs,
        low=low,
        high=high,
        kernel_size=kernel_size,
    )

    if train_mean is None or train_std is None:
        mean = emg_filtered.mean(dim=1, keepdim=True)
        std = emg_filtered.std(dim=1, keepdim=True)
    else:
        mean = train_mean.to(device=emg_filtered.device, dtype=emg_filtered.dtype)
        std = train_std.to(device=emg_filtered.device, dtype=emg_filtered.dtype)

    return (emg_filtered - mean) / (std + 1e-6)


def preprocess_imu_tensor(imu, train_mean=None, train_std=None):
    """
    Per-window Z-score normalization for IMU data.
    imu: [B, T, C]
    """
    if train_mean is not None and train_std is not None:
        mean = train_mean.to(device=imu.device, dtype=imu.dtype)
        std = train_std.to(device=imu.device, dtype=imu.dtype)
        return (imu - mean) / (std + 1e-6)

    imu_np = imu.detach().cpu().numpy().astype(np.float32)
    mean = imu_np.mean(axis=1, keepdims=True)
    std = imu_np.std(axis=1, keepdims=True)
    imu_norm = (imu_np - mean) / (std + 1e-6)
    return torch.tensor(imu_norm, dtype=imu.dtype, device=imu.device)


def compute_train_normalization_stats(dataset_subset, modality, batch_size=256):
    """
    Compute channel-wise normalization stats from train subset only.

    Returns:
      mean: [1, 1, C]
      std : [1, 1, C]
    """
    if modality not in {"emg", "imu"}:
        raise ValueError(f"Unsupported modality for normalization stats: {modality}")

    stats_loader = DataLoader(
        dataset_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    channel_sum = None
    channel_sq_sum = None
    total_count = 0

    for batch in stats_loader:
        if modality == "emg":
            _, _, _, emg, _, _ = batch
            x = filter_emg_tensor(emg).float()
        else:
            imu_acc, imu_gyr, _, _, _, _ = batch
            x = torch.cat([imu_acc, imu_gyr], dim=-1).float()

        # Aggregate over batch and time dimensions only.
        batch_sum = x.sum(dim=(0, 1))
        batch_sq_sum = (x ** 2).sum(dim=(0, 1))
        batch_count = x.shape[0] * x.shape[1]

        if channel_sum is None:
            channel_sum = batch_sum
            channel_sq_sum = batch_sq_sum
        else:
            channel_sum += batch_sum
            channel_sq_sum += batch_sq_sum

        total_count += batch_count

    if total_count == 0:
        raise RuntimeError(f"Cannot compute normalization stats for empty {modality} subset.")

    mean = channel_sum / float(total_count)
    var = channel_sq_sum / float(total_count) - (mean ** 2)
    std = torch.sqrt(torch.clamp(var, min=1e-12))

    return mean.view(1, 1, -1), std.view(1, 1, -1)


# ============================================================
# 7. Unified Conv1D + BiGRU Encoder & Models
# ============================================================
class ConvBiGRUEncoder(nn.Module):
    """
    Unified encoder for IMU, EMG, and IMU+EMG fused.
    Input  : [B, T, C]
    Output : [B, D]
    """
    def __init__(self, input_dim, d_model=128, gru_hidden=128,
                 num_layers=1, dropout=0.1):
        super().__init__()

        # ----- CNN Frontend -----
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(d_model),
            nn.Dropout(dropout),
        )

        # ----- BiGRU -----
        self.bigru = nn.GRU(
            input_size=d_model,
            hidden_size=gru_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        self.output_dim = gru_hidden * 2   # bidirectional

    def forward(self, x):
        # x: [B,T,C] → Conv: [B,d_model,T]
        x = x.permute(0, 2, 1)
        x = self.conv(x)            # [B, d_model, T]
        x = x.permute(0, 2, 1)      # [B, T, d_model]

        # BiGRU → [B,T,2H]
        outputs, _ = self.bigru(x)

        # Temporal average → [B,2H]
        feat = outputs.mean(dim=1)
        return feat


class EMGConvBiGRUModel(nn.Module):
    """
    EMG-only Conv1D+BiGRU model.
    Input : emg_x [B,T,C_emg]
    """
    def __init__(self, num_emg_channels, num_classes,
                 d_model=128, dropout=0.1):
        super().__init__()

        self.feature_dim = num_emg_channels * 4
        self.encoder = ConvBiGRUEncoder(
            input_dim=num_emg_channels,
            d_model=d_model,
            dropout=dropout,
        )
        self.feature_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fc = nn.Linear(self.encoder.output_dim + d_model, num_classes)

    def forward(self, emg_x, x_feat=None):
        feat = self.encoder(emg_x)
        if x_feat is None:
            x_feat = torch.zeros(
                feat.size(0),
                self.feature_dim,
                device=feat.device,
                dtype=feat.dtype,
            )
        feat_handcrafted = self.feature_mlp(x_feat)
        fused = torch.cat([feat, feat_handcrafted], dim=-1)
        logits = self.fc(fused)
        return logits


class SelfAttentionBlock(nn.Module):
    """
    Scaled dot-product self-attention over temporal tokens.
    Input : [B, T, D]
    Output: [B, T, D]
    """
    def __init__(self, d_model):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, v)


class EMGLSTMMSAModel(nn.Module):
    """
    EMG-only LSTM-MSA style model with dual-stage attention:
      1) attention on projected input sequence
      2) attention on BiLSTM hidden sequence

    Input : emg_x [B, T, C_emg]
    Output: logits [B, num_classes]
    """
    def __init__(
        self,
        num_emg_channels,
        num_classes,
        d_model=64,
        lstm_hidden=64,
        dropout=0.5,
    ):
        super().__init__()

        self.feature_dim = num_emg_channels * 4
        self.input_proj = nn.Linear(num_emg_channels, d_model)
        self.input_attn = SelfAttentionBlock(d_model)
        self.input_norm = nn.LayerNorm(d_model)

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        out_dim = lstm_hidden * 2
        self.output_attn = SelfAttentionBlock(out_dim)
        self.output_norm = nn.LayerNorm(out_dim)

        self.dropout = nn.Dropout(dropout)
        self.feature_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        fused_dim = d_model + out_dim + d_model
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, num_classes),
        )

    def forward(self, emg_x, x_feat=None):
        x = self.input_proj(emg_x)
        x_att = self.input_attn(x)
        x = self.input_norm(x + x_att)

        h, _ = self.lstm(x)

        h_att = self.output_attn(h)
        h = self.output_norm(h + h_att)

        # Mean pooling to summarize temporal information for both stages.
        input_summary = x.mean(dim=1)
        output_summary = h.mean(dim=1)
        if x_feat is None:
            x_feat = torch.zeros(
                input_summary.size(0),
                self.feature_dim,
                device=input_summary.device,
                dtype=input_summary.dtype,
            )
        feature_summary = self.feature_mlp(x_feat)

        fused = torch.cat([input_summary, output_summary, feature_summary], dim=-1)
        fused = self.dropout(fused)
        logits = self.classifier(fused)
        return logits


def build_class_weights_for_subset(dataset_subset, num_classes, device):
    """
    Compute inverse-frequency class weights from a torch.utils.data.Subset.
    Expects label as one-hot tensor at the last index in dataset __getitem__ output.
    """
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for sample in dataset_subset:
        one_hot = sample[-1]
        cls = int(torch.argmax(one_hot).item())
        counts[cls] += 1.0

    counts = torch.clamp(counts, min=1.0)
    inv = 1.0 / counts
    weights = inv / inv.sum() * num_classes
    return weights.to(device)


class IMUConvBiGRUModel(nn.Module):
    """
    IMU-only Conv1D+BiGRU model.
    Input : imu_x [B,T,C_imu]  (acc + gyro concatenated)
    """
    def __init__(self, num_imu_channels, num_classes,
                 d_model=128, dropout=0.1):
        super().__init__()

        self.encoder = ConvBiGRUEncoder(
            input_dim=num_imu_channels,
            d_model=d_model,
            dropout=dropout,
        )
        self.fc = nn.Linear(self.encoder.output_dim, num_classes)

    def forward(self, imu_x):
        feat = self.encoder(imu_x)
        logits = self.fc(feat)
        return logits


class IMUEMGConvBiGRUModel(nn.Module):
    """
    Early-fusion IMU+EMG Conv1D+BiGRU model.
      imu_x: [B,T,C_imu]
      emg_x: [B,T,C_emg]
      fused: [B,T,C_imu+C_emg]
    """
    def __init__(self, input_dim, num_classes,
                 d_model=128, dropout=0.1):
        super().__init__()

        self.encoder = ConvBiGRUEncoder(
            input_dim=input_dim,
            d_model=d_model,
            dropout=dropout,
        )
        self.fc = nn.Linear(self.encoder.output_dim, num_classes)

    def forward(self, emg_x, imu_x):
        fused = torch.cat([imu_x, emg_x], dim=-1)   # [B,T,C_imu+C_emg]
        feat = self.encoder(fused)
        logits = self.fc(feat)
        return logits


# ============================================================
# 8. Evaluation Helper
# ============================================================
def evaluate_model(
    model,
    test_loader,
    criterion,
    device,
    modality,
    num_emg_channels,
    num_imu_sensors,
    emg_norm_stats=None,
    imu_norm_stats=None,
):
    """
    Evaluation wrapper used for all three modalities.
    num_emg_channels / num_imu_sensors kept for API compatibility (unused).
    """
    model.eval()
    test_preds, test_true = [], []
    test_loss = 0.0
    emg_mean, emg_std = emg_norm_stats if emg_norm_stats is not None else (None, None)
    imu_mean, imu_std = imu_norm_stats if imu_norm_stats is not None else (None, None)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Testing [{modality}]"):
            if modality == "emg":
                _, _, _, emg, emg_feat, labels = batch
                emg_proc = preprocess_emg_tensor(
                    emg,
                    train_mean=emg_mean,
                    train_std=emg_std,
                ).to(device)  # [B,T,C_emg]
                emg_feat = emg_feat.to(device)
                labels = labels.to(device).argmax(dim=1)
                outputs = model(emg_proc, emg_feat)

            elif modality == "imu":
                imu_acc, imu_gyr, _, _, _, labels = batch
                imu_full = torch.cat([imu_acc, imu_gyr], dim=-1).to(device)  # [B,T,C_imu]
                imu_full = preprocess_imu_tensor(
                    imu_full,
                    train_mean=imu_mean,
                    train_std=imu_std,
                )
                labels = labels.to(device).argmax(dim=1)
                outputs = model(imu_full)

            else:  # "imu_emg"
                imu_acc, imu_gyr, _, emg, _, labels = batch
                imu_full = torch.cat([imu_acc, imu_gyr], dim=-1).to(device)  # [B,T,C_imu]
                imu_full = preprocess_imu_tensor(
                    imu_full,
                    train_mean=imu_mean,
                    train_std=imu_std,
                )

                emg_proc = preprocess_emg_tensor(
                    emg,
                    train_mean=emg_mean,
                    train_std=emg_std,
                ).to(device)  # [B,T,C_emg]
                labels = labels.to(device).argmax(dim=1)
                outputs = model(emg_proc, imu_full)

            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = outputs.max(1)
            test_preds.extend(predicted.cpu().numpy())
            test_true.extend(labels.cpu().numpy())

    test_loss /= max(len(test_loader), 1)
    acc = accuracy_score(test_true, test_preds)
    conf = confusion_matrix(test_true, test_preds, labels=list(range(len(MOVEMENT_TYPES))))
    precision, recall, f1_per_class, _ = precision_recall_fscore_support(
        test_true,
        test_preds,
        labels=list(range(len(MOVEMENT_TYPES))),
        average=None,
        zero_division=0,
    )

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        test_true,
        test_preds,
        average="macro",
        zero_division=0,
    )

    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        test_true,
        test_preds,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": acc,
        "confusion_matrix": conf,
        "precision_per_class": precision,
        "recall_per_class": recall,
        "f1_per_class": f1_per_class,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_p,
        "weighted_recall": weighted_r,
        "weighted_f1": weighted_f1,
        "test_loss": test_loss,
    }


# ============================================================
# 9. Training Setup & LOSO Loop
# ============================================================
num_epochs = 30
patience = 6
batch_size = 128
window_length = 200
window_overlap = 0

num_classes = len(MOVEMENT_TYPES)
num_emg_channels = len(config.channels_emg)
num_imu_sensors = len(config.channels_imu_acc) // 3    # kept for compatibility

# EMG model options: "convbigru" or "lstm_msa"
EMG_MODEL_VARIANT = "lstm_msa"

# EMG-targeted defaults (kept separate so IMU/IMU+EMG behavior remains stable)
EMG_D_MODEL = 128
EMG_DROPOUT = 0.2
EMG_LSTM_HIDDEN = 128
EMG_LR = 5e-4
EMG_WEIGHT_DECAY = 1e-4
EMG_USE_CLASS_WEIGHTS = True
EMG_EPOCHS = 50
EMG_PATIENCE = 12
EMG_MIN_DELTA = 0.003
EMG_LABEL_SMOOTHING = 0.1
EMG_SCHEDULER_PATIENCE = 3
EMG_GRAD_CLIP = 1.0

NON_EMG_MIN_DELTA = 0.0
NON_EMG_SCHEDULER_PATIENCE = 2
NON_EMG_GRAD_CLIP = 1.0

# IMU channel count (acc+gyr)
num_imu_channels = len(config.channels_imu_acc) + len(config.channels_imu_gyr)

# Default: 13 subjects labeled "subject_1" ... "subject_13"
if "all_subjects" not in globals():
    all_subjects = [f"subject_{i}" for i in range(1, 14)]

# Optional runtime overrides for quick smoke runs without editing source again.
SMOKE_TEST_SUBJECT = CLI_ARGS.smoke_subject or os.getenv("SMOKE_TEST_SUBJECT")
MAX_FOLDS = (
    CLI_ARGS.max_folds
    if CLI_ARGS.max_folds is not None
    else int(os.getenv("MAX_FOLDS", "0"))
)
if os.getenv("NUM_EPOCHS") is not None:
    num_epochs = int(os.getenv("NUM_EPOCHS"))
if os.getenv("BATCH_SIZE") is not None:
    batch_size = int(os.getenv("BATCH_SIZE"))
if os.getenv("WINDOW_LENGTH") is not None:
    window_length = int(os.getenv("WINDOW_LENGTH"))
if os.getenv("WINDOW_OVERLAP") is not None:
    window_overlap = int(os.getenv("WINDOW_OVERLAP"))
if os.getenv("EMG_MODEL_VARIANT") is not None:
    EMG_MODEL_VARIANT = os.getenv("EMG_MODEL_VARIANT", "lstm_msa").strip().lower()
if os.getenv("EMG_D_MODEL") is not None:
    EMG_D_MODEL = int(os.getenv("EMG_D_MODEL"))
if os.getenv("EMG_DROPOUT") is not None:
    EMG_DROPOUT = float(os.getenv("EMG_DROPOUT"))
if os.getenv("EMG_LSTM_HIDDEN") is not None:
    EMG_LSTM_HIDDEN = int(os.getenv("EMG_LSTM_HIDDEN"))
if os.getenv("EMG_LR") is not None:
    EMG_LR = float(os.getenv("EMG_LR"))
if os.getenv("EMG_WEIGHT_DECAY") is not None:
    EMG_WEIGHT_DECAY = float(os.getenv("EMG_WEIGHT_DECAY"))
if os.getenv("EMG_USE_CLASS_WEIGHTS") is not None:
    EMG_USE_CLASS_WEIGHTS = os.getenv("EMG_USE_CLASS_WEIGHTS", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
if os.getenv("EMG_EPOCHS") is not None:
    EMG_EPOCHS = int(os.getenv("EMG_EPOCHS"))
if os.getenv("EMG_PATIENCE") is not None:
    EMG_PATIENCE = int(os.getenv("EMG_PATIENCE"))
if os.getenv("EMG_MIN_DELTA") is not None:
    EMG_MIN_DELTA = float(os.getenv("EMG_MIN_DELTA"))
if os.getenv("EMG_LABEL_SMOOTHING") is not None:
    EMG_LABEL_SMOOTHING = float(os.getenv("EMG_LABEL_SMOOTHING"))
if os.getenv("EMG_SCHEDULER_PATIENCE") is not None:
    EMG_SCHEDULER_PATIENCE = int(os.getenv("EMG_SCHEDULER_PATIENCE"))
if os.getenv("EMG_GRAD_CLIP") is not None:
    EMG_GRAD_CLIP = float(os.getenv("EMG_GRAD_CLIP"))

RESULT_TAG = os.getenv("RESULT_TAG", "").strip()
if CLI_ARGS.smoke:
    RESULT_TAG = "smoke"

dataloader_workers = int(
    os.getenv("DATALOADER_WORKERS", "4" if device.type == "cuda" else "0")
)
prefetch_factor = int(os.getenv("PREFETCH_FACTOR", "2"))
persistent_workers = env_flag("PERSISTENT_WORKERS", default=(dataloader_workers > 0))

# Keep the full subject pool for training, and optionally limit only test folds.
eval_subjects = list(all_subjects)
if SMOKE_TEST_SUBJECT:
    eval_subjects = [SMOKE_TEST_SUBJECT]
elif MAX_FOLDS > 0:
    eval_subjects = eval_subjects[:MAX_FOLDS]

# Any partial run auto-tags outputs to avoid overwriting canonical 13-fold artifacts.
if not RESULT_TAG and len(eval_subjects) < len(all_subjects):
    RESULT_TAG = "smoke"

modalities = ["emg", "imu", "imu_emg"]
if os.getenv("MODALITIES"):
    requested_modalities = [m.strip() for m in os.getenv("MODALITIES", "").split(",")]
    valid_modalities = [m for m in requested_modalities if m in {"emg", "imu", "imu_emg"}]
    if valid_modalities:
        modalities = valid_modalities

results_folder = os.getenv(
    "RESULTS_FOLDER",
    "Code-base/MocapDatasetScripting_REALLAB/results/Results_ConvBiGRU",
)
os.makedirs(results_folder, exist_ok=True)

print(
    "EMG config:",
    {
        "variant": EMG_MODEL_VARIANT,
        "d_model": EMG_D_MODEL,
        "dropout": EMG_DROPOUT,
        "lstm_hidden": EMG_LSTM_HIDDEN,
        "lr": EMG_LR,
        "weight_decay": EMG_WEIGHT_DECAY,
        "class_weights": EMG_USE_CLASS_WEIGHTS,
        "epochs": EMG_EPOCHS,
        "patience": EMG_PATIENCE,
        "min_delta": EMG_MIN_DELTA,
        "label_smoothing": EMG_LABEL_SMOOTHING,
        "scheduler_patience": EMG_SCHEDULER_PATIENCE,
        "grad_clip": EMG_GRAD_CLIP,
    },
)

use_pin_memory = device.type == "cuda"
print(f"DataLoader pin_memory enabled: {use_pin_memory}")
print(
    "DataLoader config:",
    {
        "workers": dataloader_workers,
        "prefetch_factor": prefetch_factor if dataloader_workers > 0 else None,
        "persistent_workers": persistent_workers if dataloader_workers > 0 else False,
    },
)

csv_emg = os.path.join(results_folder, "Crossval_results_EMGOnly_ConvBiGRU.csv")
csv_imu = os.path.join(results_folder, "Crossval_results_IMUOnly_ConvBiGRU.csv")
csv_imu_emg = os.path.join(results_folder, "Crossval_results_IMU_EMG_ConvBiGRU.csv")
csv_combined = os.path.join(
    results_folder, f"Crossval_results_combined_EMG_{EMG_MODEL_VARIANT}.csv"
)


def tagged_path(path, tag):
    stem, ext = os.path.splitext(path)
    return f"{stem}_{tag}{ext}"


def format_lr_for_debug(lr_value):
    mantissa, exponent = f"{lr_value:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"

if RESULT_TAG:
    csv_emg = tagged_path(csv_emg, RESULT_TAG)
    csv_imu = tagged_path(csv_imu, RESULT_TAG)
    csv_imu_emg = tagged_path(csv_imu_emg, RESULT_TAG)
    csv_combined = tagged_path(csv_combined, RESULT_TAG)
    print(f"Results tag enabled: {RESULT_TAG}")

if EMG_MODEL_VARIANT == "lstm_msa":
    csv_emg = os.path.join(results_folder, "Crossval_results_EMGOnly_LSTM_MSA.csv")
    if RESULT_TAG:
        csv_emg = tagged_path(csv_emg, RESULT_TAG)

rows_emg, rows_imu, rows_imu_emg, rows_combined = [], [], [], []


for i, test_subject in enumerate(eval_subjects):
    print(f"\n====== Fold {i+1}/{len(eval_subjects)} | Test Subject: {test_subject} ======")

    train_subjects = [s for s in all_subjects if s != test_subject]

    full_train_dataset = ImuJointPairDataset(
        config,
        train_subjects,
        window_length,
        window_overlap,
        split="train",
    )
    test_dataset = ImuJointPairDataset(
        config,
        [test_subject],
        window_length,
        0,
        split="test",
    )

    val_size = int(0.2 * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    train_idx, val_idx = random_split(
        range(len(full_train_dataset)), [train_size, val_size], generator=g
    )

    train_dataset = Subset(full_train_dataset, train_idx.indices)
    val_dataset = Subset(full_train_dataset, val_idx.indices)

    need_emg_norm = any(m in {"emg", "imu_emg"} for m in modalities)
    need_imu_norm = any(m in {"imu", "imu_emg"} for m in modalities)

    emg_norm_stats = None
    imu_norm_stats = None

    if need_emg_norm:
        emg_norm_stats = compute_train_normalization_stats(train_dataset, modality="emg")
        emg_mean, emg_std = emg_norm_stats
        print(
            f"[emg] Normalization: mean={emg_mean.mean().item():.4f}, "
            f"std={emg_std.mean().item():.4f} (train-only)"
        )

    if need_imu_norm:
        imu_norm_stats = compute_train_normalization_stats(train_dataset, modality="imu")
        imu_mean, imu_std = imu_norm_stats
        print(
            f"[imu] Normalization: mean={imu_mean.mean().item():.4f}, "
            f"std={imu_std.mean().item():.4f} (train-only)"
        )

    emg_train_mean, emg_train_std = emg_norm_stats if emg_norm_stats is not None else (None, None)
    imu_train_mean, imu_train_std = imu_norm_stats if imu_norm_stats is not None else (None, None)

    common_loader_kwargs = {
        "batch_size": batch_size,
        "pin_memory": use_pin_memory,
        "num_workers": dataloader_workers,
    }
    if dataloader_workers > 0:
        common_loader_kwargs["prefetch_factor"] = prefetch_factor
        common_loader_kwargs["persistent_workers"] = persistent_workers

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **common_loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **common_loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **common_loader_kwargs,
    )

    # ---------------- Modalities Loop ----------------
    for modality in modalities:

        if modality == "emg":
            modality_num_epochs = EMG_EPOCHS
            modality_patience = EMG_PATIENCE
            modality_min_delta = EMG_MIN_DELTA
            modality_scheduler_patience = EMG_SCHEDULER_PATIENCE
            modality_grad_clip = EMG_GRAD_CLIP
            modality_label_smoothing = EMG_LABEL_SMOOTHING
            modality_lr = EMG_LR
        else:
            modality_num_epochs = num_epochs
            modality_patience = patience
            modality_min_delta = NON_EMG_MIN_DELTA
            modality_scheduler_patience = NON_EMG_SCHEDULER_PATIENCE
            modality_grad_clip = NON_EMG_GRAD_CLIP
            modality_label_smoothing = 0.0
            modality_lr = 1e-3

        print(
            f"[{modality}] Fold {i+1} config → "
            f"epochs={modality_num_epochs}, "
            f"patience={modality_patience}, "
            f"lr={format_lr_for_debug(modality_lr)}, "
            f"label_smoothing={modality_label_smoothing:.1f}"
        )

        if modality == "emg":
            if EMG_MODEL_VARIANT == "lstm_msa":
                model = EMGLSTMMSAModel(
                    num_emg_channels=num_emg_channels,
                    num_classes=num_classes,
                    d_model=EMG_D_MODEL,
                    lstm_hidden=EMG_LSTM_HIDDEN,
                    dropout=EMG_DROPOUT,
                ).to(device)
            else:
                model = EMGConvBiGRUModel(
                    num_emg_channels=num_emg_channels,
                    num_classes=num_classes,
                    d_model=EMG_D_MODEL,
                    dropout=EMG_DROPOUT,
                ).to(device)
            model_path = os.path.join(
                results_folder,
                f"subject_{i+1}_EMGOnly_{EMG_MODEL_VARIANT}.pt",
            )
            if RESULT_TAG:
                model_path = tagged_path(model_path, RESULT_TAG)

        elif modality == "imu":
            model = IMUConvBiGRUModel(
                num_imu_channels=num_imu_channels,
                num_classes=num_classes,
                d_model=128,
                dropout=0.1,
            ).to(device)
            model_path = os.path.join(
                results_folder, f"subject_{i+1}_IMUOnly_ConvBiGRU.pt"
            )
            if RESULT_TAG:
                model_path = tagged_path(model_path, RESULT_TAG)

        else:  # "imu_emg"
            total_input_dim = num_imu_channels + num_emg_channels
            model = IMUEMGConvBiGRUModel(
                input_dim=total_input_dim,
                num_classes=num_classes,
                d_model=128,
                dropout=0.1,
            ).to(device)
            model_path = os.path.join(
                results_folder, f"subject_{i+1}_IMU_EMG_ConvBiGRU.pt"
            )
            if RESULT_TAG:
                model_path = tagged_path(model_path, RESULT_TAG)

        if modality == "emg":
            optimizer = optim.AdamW(
                model.parameters(),
                lr=EMG_LR,
                weight_decay=EMG_WEIGHT_DECAY,
            )
            if EMG_USE_CLASS_WEIGHTS:
                class_weights = build_class_weights_for_subset(
                    train_dataset,
                    num_classes=num_classes,
                    device=device,
                )
                criterion = nn.CrossEntropyLoss(
                    weight=class_weights,
                    label_smoothing=modality_label_smoothing,
                )
            else:
                criterion = nn.CrossEntropyLoss(label_smoothing=modality_label_smoothing)
        else:
            optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
            criterion = nn.CrossEntropyLoss(label_smoothing=modality_label_smoothing)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=modality_scheduler_patience,
        )

        best_val_loss = float("inf")
        patience_counter = 0

        # ---------------- Training ----------------
        for epoch in range(modality_num_epochs):
            model.train()
            running_loss = 0.0

            for batch in tqdm(
                train_loader,
                desc=f"[{modality}] Epoch {epoch+1}/{modality_num_epochs}",
            ):
                if modality == "emg":
                    _, _, _, emg, emg_feat, labels = batch
                    emg_proc = preprocess_emg_tensor(
                        emg,
                        train_mean=emg_train_mean,
                        train_std=emg_train_std,
                    ).to(device)  # [B,T,C_emg]
                    emg_feat = emg_feat.to(device)
                    labels = labels.to(device).argmax(dim=1)
                    outputs = model(emg_proc, emg_feat)

                elif modality == "imu":
                    imu_acc, imu_gyr, _, _, _, labels = batch
                    imu_full = torch.cat([imu_acc, imu_gyr], dim=-1).to(device)
                    imu_full = preprocess_imu_tensor(
                        imu_full,
                        train_mean=imu_train_mean,
                        train_std=imu_train_std,
                    )
                    labels = labels.to(device).argmax(dim=1)
                    outputs = model(imu_full)

                else:  # "imu_emg"
                    imu_acc, imu_gyr, _, emg, _, labels = batch
                    imu_full = torch.cat([imu_acc, imu_gyr], dim=-1).to(device)
                    imu_full = preprocess_imu_tensor(
                        imu_full,
                        train_mean=imu_train_mean,
                        train_std=imu_train_std,
                    )

                    emg_proc = preprocess_emg_tensor(
                        emg,
                        train_mean=emg_train_mean,
                        train_std=emg_train_std,
                    ).to(device)
                    labels = labels.to(device).argmax(dim=1)
                    outputs = model(emg_proc, imu_full)

                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), modality_grad_clip)
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / max(len(train_loader), 1)

            # ---------------- Validation ----------------
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    if modality == "emg":
                        _, _, _, emg, emg_feat, labels = batch
                        emg_proc = preprocess_emg_tensor(
                            emg,
                            train_mean=emg_train_mean,
                            train_std=emg_train_std,
                        ).to(device)
                        emg_feat = emg_feat.to(device)
                        labels = labels.to(device).argmax(dim=1)
                        outputs = model(emg_proc, emg_feat)

                    elif modality == "imu":
                        imu_acc, imu_gyr, _, _, _, labels = batch
                        imu_full = torch.cat([imu_acc, imu_gyr], dim=-1).to(device)
                        imu_full = preprocess_imu_tensor(
                            imu_full,
                            train_mean=imu_train_mean,
                            train_std=imu_train_std,
                        )
                        labels = labels.to(device).argmax(dim=1)
                        outputs = model(imu_full)

                    else:
                        imu_acc, imu_gyr, _, emg, _, labels = batch
                        imu_full = torch.cat([imu_acc, imu_gyr], dim=-1).to(device)
                        imu_full = preprocess_imu_tensor(
                            imu_full,
                            train_mean=imu_train_mean,
                            train_std=imu_train_std,
                        )
                        emg_proc = preprocess_emg_tensor(
                            emg,
                            train_mean=emg_train_mean,
                            train_std=emg_train_std,
                        ).to(device)
                        labels = labels.to(device).argmax(dim=1)
                        outputs = model(emg_proc, imu_full)

                    val_loss += criterion(outputs, labels).item()

            val_loss /= max(len(val_loader), 1)
            scheduler.step(val_loss)
            print(
                f"[{modality}] Epoch {epoch+1} | "
                f"Train: {avg_loss:.4f} | Val: {val_loss:.4f}"
            )

            # Early stopping
            if val_loss < (best_val_loss - modality_min_delta):
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), model_path)
            else:
                patience_counter += 1
                if patience_counter >= modality_patience:
                    print(f"[{modality}] Early stopping.")
                    break

        print(f"[{modality}] ✅ Best model saved: {model_path}")

        # ---------------- Test ----------------
        model.load_state_dict(torch.load(model_path, map_location=device))

        metrics = evaluate_model(
            model,
            test_loader,
            criterion,
            device,
            modality,
            num_emg_channels=num_emg_channels,
            num_imu_sensors=num_imu_sensors,
            emg_norm_stats=(emg_train_mean, emg_train_std),
            imu_norm_stats=(imu_train_mean, imu_train_std),
        )

        row = {
            "fold": i + 1,
            "test_subject": test_subject,
            "emg_model_variant": EMG_MODEL_VARIANT if modality == "emg" else "n/a",
            "accuracy": metrics["accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "weighted_precision": metrics["weighted_precision"],
            "weighted_recall": metrics["weighted_recall"],
            "weighted_f1": metrics["weighted_f1"],
            "test_loss": metrics["test_loss"],
            "precision_per_class": np.array_str(metrics["precision_per_class"]),
            "recall_per_class": np.array_str(metrics["recall_per_class"]),
            "f1_per_class": np.array_str(metrics["f1_per_class"]),
            "confusion_matrix": metrics["confusion_matrix"].tolist(),
        }

        if modality == "emg":
            rows_emg.append(row)
            pd.DataFrame(rows_emg).to_csv(csv_emg, index=False)
        elif modality == "imu":
            rows_imu.append(row)
            pd.DataFrame(rows_imu).to_csv(csv_imu, index=False)
        else:
            rows_imu_emg.append(row)
            pd.DataFrame(rows_imu_emg).to_csv(csv_imu_emg, index=False)

    # ---------------- Combined Summary Row ----------------
    last_emg = rows_emg[-1] if rows_emg else None
    last_imu = rows_imu[-1] if rows_imu else None
    last_imu_emg = rows_imu_emg[-1] if rows_imu_emg else None

    rows_combined.append(
        {
            "fold": i + 1,
            "test_subject": test_subject,
            "emg_model_variant": EMG_MODEL_VARIANT,
            "acc_emg": last_emg["accuracy"] if last_emg else None,
            "acc_imu": last_imu["accuracy"] if last_imu else None,
            "acc_imu_emg": last_imu_emg["accuracy"] if last_imu_emg else None,
            "macro_f1_emg": last_emg["macro_f1"] if last_emg else None,
            "macro_f1_imu": last_imu["macro_f1"] if last_imu else None,
            "macro_f1_imu_emg": last_imu_emg["macro_f1"] if last_imu_emg else None,
        }
    )
    pd.DataFrame(rows_combined).to_csv(csv_combined, index=False)
    print(f"📄 Combined results updated: {csv_combined}")

    # Optional zip archive after each subject
    cumulative_zip_path = os.getenv(
        "RESULTS_ZIP_PATH",
        "Code-base/MocapDatasetScripting_REALLAB/results/ConvBiGRU_results.zip",
    )
    if RESULT_TAG:
        cumulative_zip_path = tagged_path(cumulative_zip_path, RESULT_TAG)
    os.makedirs(os.path.dirname(cumulative_zip_path), exist_ok=True)
    shutil.make_archive(
        base_name=cumulative_zip_path.replace(".zip", ""),
        format="zip",
        root_dir=results_folder,
    )
    print(f"✅ Cumulative ZIP updated → {cumulative_zip_path}")

print("\n🎉 All folds completed!")
print(f"📌 Final combined CSV: {csv_combined}")


def summarize_emg_variant_vs_baseline(
    results_dir,
    current_emg_csv,
    baseline_emg_csv,
    result_tag="",
):
    """Create a compact comparison CSV between baseline EMG ConvBiGRU and current EMG variant."""
    if not (os.path.exists(current_emg_csv) and os.path.exists(baseline_emg_csv)):
        print("ℹ️ Baseline/current EMG CSV missing; skipping comparison table.")
        return

    current_df = pd.read_csv(current_emg_csv)
    baseline_df = pd.read_csv(baseline_emg_csv)

    def parse_vector_str(vec_str):
        if pd.isna(vec_str):
            return np.array([])
        clean = str(vec_str).strip().strip("[").strip("]")
        if not clean:
            return np.array([])
        return np.fromstring(clean, sep=" ")

    def ensure_macro_f1(df):
        if "macro_f1" in df.columns:
            return df

        if {"precision_per_class", "recall_per_class"}.issubset(df.columns):
            macro_f1_vals = []
            for _, row in df.iterrows():
                p = parse_vector_str(row["precision_per_class"])
                r = parse_vector_str(row["recall_per_class"])
                if p.size == 0 or r.size == 0 or p.size != r.size:
                    macro_f1_vals.append(np.nan)
                    continue

                denom = p + r
                f1 = np.where(denom > 0, 2.0 * p * r / denom, 0.0)
                macro_f1_vals.append(float(np.mean(f1)))

            df = df.copy()
            df["macro_f1"] = macro_f1_vals
            return df

        df = df.copy()
        df["macro_f1"] = np.nan
        return df

    current_df = ensure_macro_f1(current_df)
    baseline_df = ensure_macro_f1(baseline_df)

    required_cols = {"test_subject", "accuracy", "macro_f1"}
    if not (required_cols.issubset(current_df.columns) and required_cols.issubset(baseline_df.columns)):
        print("ℹ️ Required columns missing for comparison; skipping comparison table.")
        return

    merged = pd.merge(
        baseline_df[["test_subject", "accuracy", "macro_f1"]],
        current_df[["test_subject", "accuracy", "macro_f1"]],
        on="test_subject",
        how="inner",
        suffixes=("_baseline", "_current"),
    )

    if merged.empty:
        print("ℹ️ No overlapping subjects between baseline and current results.")
        return

    merged["delta_accuracy"] = merged["accuracy_current"] - merged["accuracy_baseline"]
    merged["delta_macro_f1"] = merged["macro_f1_current"] - merged["macro_f1_baseline"]

    comparison_detail_csv = os.path.join(
        results_dir,
        f"EMG_baseline_vs_{EMG_MODEL_VARIANT}_by_subject.csv",
    )
    if result_tag:
        stem, ext = os.path.splitext(comparison_detail_csv)
        comparison_detail_csv = f"{stem}_{result_tag}{ext}"
    merged.to_csv(comparison_detail_csv, index=False)

    summary_df = pd.DataFrame(
        [
            {
                "model": "EMG_ConvBiGRU_baseline",
                "mean_accuracy": merged["accuracy_baseline"].mean(),
                "mean_macro_f1": merged["macro_f1_baseline"].mean(),
            },
            {
                "model": f"EMG_{EMG_MODEL_VARIANT}_current",
                "mean_accuracy": merged["accuracy_current"].mean(),
                "mean_macro_f1": merged["macro_f1_current"].mean(),
            },
            {
                "model": "delta_current_minus_baseline",
                "mean_accuracy": merged["delta_accuracy"].mean(),
                "mean_macro_f1": merged["delta_macro_f1"].mean(),
            },
        ]
    )

    comparison_summary_csv = os.path.join(
        results_dir,
        f"EMG_baseline_vs_{EMG_MODEL_VARIANT}_summary.csv",
    )
    if result_tag:
        stem, ext = os.path.splitext(comparison_summary_csv)
        comparison_summary_csv = f"{stem}_{result_tag}{ext}"
    summary_df.to_csv(comparison_summary_csv, index=False)

    print("\n📊 EMG comparison summary (baseline vs current):")
    print(summary_df.to_string(index=False))
    print(f"📄 Subject-wise comparison saved: {comparison_detail_csv}")
    print(f"📄 Summary comparison saved: {comparison_summary_csv}")


summarize_emg_variant_vs_baseline(
    results_folder,
    csv_emg,
    os.path.join(results_folder, "Crossval_results_EMGOnly_ConvBiGRU.csv"),
    result_tag=RESULT_TAG,
)
