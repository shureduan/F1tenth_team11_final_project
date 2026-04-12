import numpy as np
import pandas as pd
import torch
from pathlib import Path


def minmax_01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Column-wise min-max normalization to [0, 1].
    Constant columns are mapped to 0.
    """
    x = np.asarray(x, dtype=np.float32)
    x_min = x.min(axis=0)
    x_max = x.max(axis=0)
    denom = np.maximum(x_max - x_min, eps)
    x_norm = (x - x_min) / denom
    const_mask = (x_max - x_min) < eps
    if np.any(const_mask):
        x_norm[:, const_mask] = 0.0
    return x_norm


def compute_d_prior_from_local_csv(local_csv_path: str) -> pd.DataFrame:
    """
    Read x_local_features_u.csv and compute a local-geometry prior D_prior_local in [-1, 1].

    Required columns:
        segment_id, kappa, delta_kappa, delta_psi, d_left, d_right, d_inner

    Logic:
        Positive tendency (more defend-apex):
            - larger |kappa|
            - larger |delta_kappa|
            - larger |delta_psi|
            - smaller d_inner

        Negative tendency (more best-line / neutral):
            - larger d_inner
    """
    df = pd.read_csv(local_csv_path)

    required = ["segment_id", "kappa", "delta_kappa", "delta_psi", "d_left", "d_right", "d_inner"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Use absolute values for geometric intensity
    local_raw = np.column_stack([
        np.abs(df["kappa"].to_numpy(dtype=np.float32)),
        np.abs(df["delta_kappa"].to_numpy(dtype=np.float32)),
        np.abs(df["delta_psi"].to_numpy(dtype=np.float32)),
        df["d_inner"].to_numpy(dtype=np.float32),
    ])

    # Normalize each feature to [0, 1]
    local_norm = minmax_01(local_raw)

    kappa_n = local_norm[:, 0]
    delta_kappa_n = local_norm[:, 1]
    delta_psi_n = local_norm[:, 2]
    d_inner_n = local_norm[:, 3]

    # Concrete prior score from local geometry only
    raw_score = (
        0.40 * kappa_n
        + 0.20 * delta_kappa_n
        + 0.20 * delta_psi_n
        + 0.20 * (1.0 - d_inner_n)
    )

    # Center around zero, then bound to [-1, 1]
    raw_centered = raw_score - np.median(raw_score)
    d_prior_local = np.tanh(2.0 * raw_centered).astype(np.float32)

    out_df = pd.DataFrame({
        "segment_id": df["segment_id"].to_numpy(dtype=int),
        "abs_kappa_01": kappa_n,
        "abs_delta_kappa_01": delta_kappa_n,
        "abs_delta_psi_01": delta_psi_n,
        "inv_d_inner_01": 1.0 - d_inner_n,
        "D_prior_local": d_prior_local,
    })
    return out_df


def l_prior_loss(d_pred: torch.Tensor, d_prior: torch.Tensor) -> torch.Tensor:
    """
    Prior consistency loss:
        L_prior = mean( (D_pred - D_prior)^2 )

    Shapes:
        d_pred  : [B, N] or [N]
        d_prior : [B, N] or [N]
    """
    return torch.mean((d_pred - d_prior) ** 2)


if __name__ == "__main__":
    LOCAL_CSV = "/Users/shure_duan/VScode/f1tenth/neural_input/local_input/local_result/x_local_features_u.csv"
    OUT_CSV = "/Users/shure_duan/VScode/f1tenth/neural_network/output_u/d_prior_from_local_u.csv"

    out_df = compute_d_prior_from_local_csv(LOCAL_CSV)
    Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)

    print("Saved:", OUT_CSV)
    print(out_df.head())
