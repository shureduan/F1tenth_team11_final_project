
import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================================================
# Fixed paths (user-provided)
# =========================================================
TRACK_NAME = "Shanghai"

HEURISTIC_CSV = Path("/Users/shure_duan/VScode/f1tenth/neural_input/heuristic_score_input/track_heuristic_scores_u.csv")
X_LOCAL_CSV = Path("/Users/shure_duan/VScode/f1tenth/neural_input/local_input/local_result/x_local_features_u.csv")
X_CONTEXT_CSV = Path("/Users/shure_duan/VScode/f1tenth/neural_input/look_ahead_context_input/look_ahead_result/x_context_features_u.csv")
LABEL_DIR = Path("/Users/shure_duan/VScode/f1tenth/neural_network/label_data")

OUT_DIR = Path("/Users/shure_duan/VScode/f1tenth/neural_network/output_u")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUT_DIR / "defense_preference_cnn_mlp_u.pt"
PRED_CSV = OUT_DIR / "defense_preference_prediction_u.csv"
NORM_JSON = OUT_DIR / "normalization_params_u.json"


# =========================================================
# Hyperparameters
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 300
LR = 1e-3
WEIGHT_DECAY = 1e-5

LAMBDA_STRATEGY = 1.0
LAMBDA_SMOOTH = 0.10
LAMBDA_PRIOR = 0.20

CNN_HIDDEN = 8
MLP_HIDDEN = 8
KERNEL_SIZE = 5


# =========================================================
# Utility functions
# =========================================================
def minmax_normalize(x: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    """
    Column-wise min-max normalization to [0, 1].
    Constant columns are mapped to 0.
    """
    x = np.asarray(x, dtype=np.float32)
    x_min = x.min(axis=0)
    x_max = x.max(axis=0)
    denom = np.maximum(x_max - x_min, eps)
    x_norm = (x - x_min) / denom
    # if constant column, set to zero explicitly
    const_mask = (x_max - x_min) < eps
    if np.any(const_mask):
        x_norm[:, const_mask] = 0.0
    params = {
        "min": x_min.tolist(),
        "max": x_max.tolist(),
    }
    return x_norm.astype(np.float32), params


def repeat_prior(prior_vec: np.ndarray, n_segments: int) -> np.ndarray:
    prior_vec = np.asarray(prior_vec, dtype=np.float32).reshape(1, -1)
    return np.repeat(prior_vec, n_segments, axis=0)


def wrap_find_label_file(label_dir: Path, track_name: str) -> Path:
    """
    Find one label file for the given track name.
    Supported:
      - *.csv with one column in {D_target, target, label, score}
      - *.npy
      - *.pt / *.pth containing a 1D tensor/array
    """
    patterns = [
        f"{track_name}*.csv",
        f"{track_name}*.npy",
        f"{track_name}*.pt",
        f"{track_name}*.pth",
        f"*{track_name}*.csv",
        f"*{track_name}*.npy",
        f"*{track_name}*.pt",
        f"*{track_name}*.pth",
    ]
    matches = []
    for pat in patterns:
        matches.extend(sorted(label_dir.glob(pat)))
    if not matches:
        raise FileNotFoundError(
            f"No label file found in {label_dir} for track '{track_name}'. "
            f"Please place one file like {track_name}_label.csv or {track_name}_target.npy there."
        )
    return matches[0]


def load_target_vector(label_path: Path, expected_len: int) -> np.ndarray:
    if label_path.suffix.lower() == ".csv":
        df = pd.read_csv(label_path)
        candidate_cols = ["D_target", "target", "label", "score", "D_map"]
        col = None
        for c in candidate_cols:
            if c in df.columns:
                col = c
                break
        if col is None:
            # fallback: if exactly one numeric column, use it
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if len(numeric_cols) == 1:
                col = numeric_cols[0]
            else:
                raise ValueError(
                    f"CSV label file {label_path} must contain one of {candidate_cols}, "
                    f"or exactly one numeric column."
                )
        y = df[col].to_numpy(dtype=np.float32)

    elif label_path.suffix.lower() == ".npy":
        y = np.load(label_path).astype(np.float32).reshape(-1)

    elif label_path.suffix.lower() in [".pt", ".pth"]:
        obj = torch.load(label_path, map_location="cpu")
        if isinstance(obj, dict):
            # common keys
            for k in ["D_target", "target", "label", "score"]:
                if k in obj:
                    obj = obj[k]
                    break
        if isinstance(obj, torch.Tensor):
            y = obj.detach().cpu().numpy().astype(np.float32).reshape(-1)
        else:
            y = np.asarray(obj, dtype=np.float32).reshape(-1)
    else:
        raise ValueError(f"Unsupported label file type: {label_path.suffix}")

    if len(y) != expected_len:
        raise ValueError(
            f"Label length mismatch: expected {expected_len}, got {len(y)} from {label_path}"
        )
    return y


# =========================================================
# Feature loading
# =========================================================
def load_local_features(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    required = ["segment_id", "kappa", "delta_kappa", "delta_psi", "d_left", "d_right", "d_inner"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"{path} missing required column: {c}")

    x_local = df[["kappa", "delta_kappa", "delta_psi", "d_left", "d_right", "d_inner"]].to_numpy(dtype=np.float32)
    segment_id = df["segment_id"].to_numpy(dtype=np.int64)
    return x_local, segment_id


def load_context_features(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    required = [
        "mean_curvature_30",
        "max_curvature_30",
        "straight_indicator_30",
        "compound_indicator_30",
        "accum_heading_change_30",
    ]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"{path} missing required column: {c}")

    x_context = df[required].to_numpy(dtype=np.float32)
    return x_context


def load_prior_vector(path: Path, track_name: str) -> np.ndarray:
    df = pd.read_csv(path)
    required = ["track_name", "layout_style_score", "overtaking_friendliness_score", "driver_challenge_score"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"{path} missing required column: {c}")

    row = df[df["track_name"].astype(str) == track_name]
    if row.empty:
        raise ValueError(f"Track '{track_name}' not found in {path}")
    prior = row.iloc[0][["layout_style_score", "overtaking_friendliness_score", "driver_challenge_score"]].to_numpy(dtype=np.float32)
    return prior


# =========================================================
# D_prior construction
# =========================================================
def build_d_prior(
    x_local_norm: np.ndarray,
    x_context_norm: np.ndarray,
    x_prior_repeat_norm: np.ndarray,
) -> np.ndarray:
    """
    Build a weak heuristic target D_prior(s_i) in [-1, 1].

    Positive tendency:
      - larger current curvature
      - larger future curvature
      - stronger compound-turn pattern
      - smaller inner-side space
      - higher driver challenge

    Negative tendency:
      - stronger straight indicator
      - stronger straight-line / overtaking style prior
    """
    kappa = x_local_norm[:, 0]
    d_inner = x_local_norm[:, 5]

    mean_curv = x_context_norm[:, 0]
    max_curv = x_context_norm[:, 1]
    straight_ind = x_context_norm[:, 2]
    compound_ind = x_context_norm[:, 3]

    layout_prior = x_prior_repeat_norm[:, 0]      # 1~5 -> [0,1]
    overtake_prior = x_prior_repeat_norm[:, 1]    # 1~5 -> [0,1]
    challenge_prior = x_prior_repeat_norm[:, 2]   # 1~5 -> [0,1]

    corner_strength = 0.35 * kappa + 0.20 * mean_curv + 0.20 * max_curv + 0.15 * compound_ind
    apex_pressure = 0.15 * (1.0 - d_inner) + 0.10 * challenge_prior
    straight_penalty = 0.20 * straight_ind + 0.10 * layout_prior + 0.05 * overtake_prior

    raw = corner_strength + apex_pressure - straight_penalty
    # map approximately to [-1, 1]
    d_prior = np.tanh(2.0 * (raw - 0.5)).astype(np.float32)
    return d_prior


# =========================================================
# Dataset
# =========================================================
class SingleTrackDataset(Dataset):
    """
    One whole track = one training sample
    X_seq:   [600, 11]
    x_prior: [3]
    y:       [600]
    d_prior: [600]
    """
    def __init__(
        self,
        x_seq: np.ndarray,
        x_prior: np.ndarray,
        y: np.ndarray,
        d_prior: np.ndarray,
    ) -> None:
        self.x_seq = torch.from_numpy(x_seq).float()
        self.x_prior = torch.from_numpy(x_prior).float()
        self.y = torch.from_numpy(y).float()
        self.d_prior = torch.from_numpy(d_prior).float()

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int):
        return self.x_seq, self.x_prior, self.y, self.d_prior


# =========================================================
# Model
# =========================================================
class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int = 11, hidden_channels: int = 8, kernel_size: int = 5):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C] -> [B, C, N]
        x = x.transpose(1, 2)
        z = self.net(x)
        # [B, C, N] -> [B, N, C]
        z = z.transpose(1, 2)
        return z


class MLPHead(nn.Module):
    def __init__(self, in_dim: int = 11, hidden_dim: int = 8, use_tanh: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.use_tanh = use_tanh

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, N, in_dim]
        out = self.fc1(z)
        out = self.act(out)
        out = self.fc2(out)
        if self.use_tanh:
            out = torch.tanh(out)
        return out.squeeze(-1)  # [B, N]


class DefensePreferenceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNNEncoder(in_channels=11, hidden_channels=CNN_HIDDEN, kernel_size=KERNEL_SIZE)
        self.head = MLPHead(in_dim=CNN_HIDDEN + 3, hidden_dim=MLP_HIDDEN, use_tanh=True)

    def forward(self, x_seq: torch.Tensor, x_prior: torch.Tensor) -> torch.Tensor:
        """
        x_seq:   [B, 600, 11]
        x_prior: [B, 3]
        """
        z_cnn = self.cnn(x_seq)                                # [B, 600, 8]
        x_prior_repeat = x_prior.unsqueeze(1).repeat(1, z_cnn.size(1), 1)  # [B, 600, 3]
        z = torch.cat([z_cnn, x_prior_repeat], dim=-1)        # [B, 600, 11]
        d_map = self.head(z)                                   # [B, 600]
        return d_map


# =========================================================
# Losses
# =========================================================
def strategy_loss(d_pred: torch.Tensor, d_target: torch.Tensor) -> torch.Tensor:
    return torch.mean((d_pred - d_target) ** 2)


def smoothness_loss(d_pred: torch.Tensor) -> torch.Tensor:
    # First-order smoothness
    return torch.mean((d_pred[:, 1:] - d_pred[:, :-1]) ** 2)


def prior_consistency_loss(d_pred: torch.Tensor, d_prior: torch.Tensor) -> torch.Tensor:
    return torch.mean((d_pred - d_prior) ** 2)


# =========================================================
# Main
# =========================================================
def main():
    # -----------------------------
    # Load features
    # -----------------------------
    x_local_raw, segment_id = load_local_features(X_LOCAL_CSV)
    x_context_raw = load_context_features(X_CONTEXT_CSV)
    x_prior_raw = load_prior_vector(HEURISTIC_CSV, TRACK_NAME)

    n_segments = x_local_raw.shape[0]
    if x_context_raw.shape[0] != n_segments:
        raise ValueError(f"Segment count mismatch: local={n_segments}, context={x_context_raw.shape[0]}")

    # -----------------------------
    # Normalize X_local and X_context to [0, 1]
    # -----------------------------
    x_local_norm, local_norm_params = minmax_normalize(x_local_raw)
    x_context_norm, context_norm_params = minmax_normalize(x_context_raw)

    # Prior uses simple 1~5 -> [0,1]
    x_prior_norm = ((x_prior_raw - 1.0) / 4.0).astype(np.float32)

    # Concatenate sequence features
    x_seq = np.concatenate([x_local_norm, x_context_norm], axis=1).astype(np.float32)   # [600, 11]

    # -----------------------------
    # Load labels
    # -----------------------------
    label_file = wrap_find_label_file(LABEL_DIR, TRACK_NAME)
    y_target = load_target_vector(label_file, expected_len=n_segments).astype(np.float32)

    # -----------------------------
    # Build D_prior
    # -----------------------------
    x_prior_repeat_norm = repeat_prior(x_prior_norm, n_segments)
    d_prior = build_d_prior(x_local_norm, x_context_norm, x_prior_repeat_norm)

    # -----------------------------
    # Save normalization info
    # -----------------------------
    norm_info = {
        "track_name": TRACK_NAME,
        "local_feature_order": ["kappa", "delta_kappa", "delta_psi", "d_left", "d_right", "d_inner"],
        "context_feature_order": [
            "mean_curvature_30",
            "max_curvature_30",
            "straight_indicator_30",
            "compound_indicator_30",
            "accum_heading_change_30",
        ],
        "prior_feature_order": [
            "layout_style_score",
            "overtaking_friendliness_score",
            "driver_challenge_score",
        ],
        "local_minmax": local_norm_params,
        "context_minmax": context_norm_params,
        "prior_raw": x_prior_raw.tolist(),
        "prior_norm_rule": "x_prior_norm = (x_prior_raw - 1.0) / 4.0",
        "label_file": str(label_file),
    }
    with open(NORM_JSON, "w", encoding="utf-8") as f:
        json.dump(norm_info, f, indent=2)

    # -----------------------------
    # Dataset / loader
    # -----------------------------
    dataset = SingleTrackDataset(x_seq=x_seq, x_prior=x_prior_norm, y=y_target, d_prior=d_prior)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # -----------------------------
    # Model / optimizer
    # -----------------------------
    model = DefensePreferenceNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # -----------------------------
    # Train
    # -----------------------------
    model.train()
    for epoch in range(1, EPOCHS + 1):
        for x_seq_b, x_prior_b, y_b, d_prior_b in loader:
            x_seq_b = x_seq_b.to(DEVICE)
            x_prior_b = x_prior_b.to(DEVICE)
            y_b = y_b.to(DEVICE)
            d_prior_b = d_prior_b.to(DEVICE)

            d_pred = model(x_seq_b, x_prior_b)

            loss_strategy = strategy_loss(d_pred, y_b)
            loss_smooth = smoothness_loss(d_pred)
            loss_prior = prior_consistency_loss(d_pred, d_prior_b)

            loss = (
                LAMBDA_STRATEGY * loss_strategy
                + LAMBDA_SMOOTH * loss_smooth
                + LAMBDA_PRIOR * loss_prior
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch == 1 or epoch % 20 == 0 or epoch == EPOCHS:
            print(
                f"Epoch {epoch:03d} | "
                f"L_total={loss.item():.6f} | "
                f"L_strategy={loss_strategy.item():.6f} | "
                f"L_smooth={loss_smooth.item():.6f} | "
                f"L_prior={loss_prior.item():.6f}"
            )

    # -----------------------------
    # Save model
    # -----------------------------
    torch.save(
        {
            "track_name": TRACK_NAME,
            "model_state_dict": model.state_dict(),
            "hyperparameters": {
                "epochs": EPOCHS,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "lambda_strategy": LAMBDA_STRATEGY,
                "lambda_smooth": LAMBDA_SMOOTH,
                "lambda_prior": LAMBDA_PRIOR,
                "cnn_hidden": CNN_HIDDEN,
                "mlp_hidden": MLP_HIDDEN,
                "kernel_size": KERNEL_SIZE,
            },
        },
        MODEL_PATH,
    )

    # -----------------------------
    # Predict and save
    # -----------------------------
    model.eval()
    with torch.no_grad():
        x_seq_t = torch.from_numpy(x_seq).unsqueeze(0).float().to(DEVICE)
        x_prior_t = torch.from_numpy(x_prior_norm).unsqueeze(0).float().to(DEVICE)
        d_pred = model(x_seq_t, x_prior_t).squeeze(0).cpu().numpy()

    out_df = pd.DataFrame({
        "segment_id": segment_id,
        "D_pred": d_pred,
        "D_target": y_target,
        "D_prior": d_prior,
    })
    out_df.to_csv(PRED_CSV, index=False)

    print("\nDone.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Prediction CSV saved to: {PRED_CSV}")
    print(f"Normalization params saved to: {NORM_JSON}")


if __name__ == "__main__":
    main()
