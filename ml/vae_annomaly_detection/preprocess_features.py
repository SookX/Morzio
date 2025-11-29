from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml


DEFAULT_FEATURE_COLUMNS: List[str] = [
    "estimated_monthly_income",
    "last_inflow_amount",
    "days_since_last_inflow",
    "credit_score",
    "total_spend_30d",
    "total_spend_90d",
    "transaction_count_30d",
    "transaction_count_90d",
    "avg_txn_amount_30d",
    "avg_txn_amount_90d",
    "max_txn_amount_90d",
    "txn_amount_median_90d",
    "spend_volatility_30d",
    "spend_volatility_90d",
    "spend_to_income_ratio_30d",
    "spend_to_income_ratio_90d",
    "avg_txn_over_income_ratio_90d",
    "txn_count_30d_norm",
    "current_txn_amount",
    "current_txn_mcc",
]

# Feature weights to emphasize certain signals (higher weight = higher impact after scaling)
DEFAULT_FEATURE_WEIGHTS: Dict[str, float] = {
    # Affordability signals
    "spend_to_income_ratio_30d": 2.0,
    "spend_to_income_ratio_90d": 2.0,
    "avg_txn_over_income_ratio_90d": 1.5,
    "current_txn_amount": 1.5,
    # Counts and volatility left at 1.0 by omission
}


def manual_standard_scale(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return standardized array, means, and stds (stds clamped to 1e-9 minimum)."""
    means = arr.mean(axis=0)
    stds = arr.std(axis=0)
    stds = np.where(stds < 1e-9, 1.0, stds)
    scaled = (arr - means) / stds
    return scaled, means, stds


def apply_feature_weights(arr: np.ndarray, columns: List[str], weights: Dict[str, float]) -> np.ndarray:
    """Multiply columns by provided weights (defaults to 1.0 if not specified)."""
    for idx, col in enumerate(columns):
        w = weights.get(col, 1.0)
        if w != 1.0:
            arr[:, idx] *= w
    return arr


def split_data(arr: np.ndarray, train_frac: float, val_frac: float, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split into train/val/test by fractions."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(arr))
    rng.shuffle(idx)
    n = len(arr)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return arr[train_idx], arr[val_idx], arr[test_idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess training features for VAE (scaling + weighting).")
    parser.add_argument("--input", type=Path, default=Path("ml/dema_revenue/output/training_features.csv"), help="Path to training features CSV.")
    parser.add_argument("--output", type=Path, default=Path("ml/vae_annomaly_detection/output/processed_features.npz"), help="Output NPZ path.")
    parser.add_argument("--config", type=Path, default=Path("ml/vae_annomaly_detection/config.yaml"), help="Config YAML for split ratios.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    # Drop id/time columns if present
    feature_df = df[[c for c in DEFAULT_FEATURE_COLUMNS if c in df.columns]].copy()
    if feature_df.isnull().values.any():
        feature_df = feature_df.dropna()

    data = feature_df.to_numpy(dtype=float)

    scaled, means, stds = manual_standard_scale(data)
    weighted = apply_feature_weights(scaled, list(feature_df.columns), DEFAULT_FEATURE_WEIGHTS)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    train_frac = cfg.get("data", {}).get("train_split", 0.8)
    val_frac = cfg.get("data", {}).get("validation_split", 0.1)
    train, val, test = split_data(weighted, train_frac, val_frac, seed=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        train=train,
        val=val,
        test=test,
        columns=np.array(feature_df.columns),
        means=means,
        stds=stds,
    )
    scaler_meta = {
        "columns": list(feature_df.columns),
        "means": means.tolist(),
        "stds": stds.tolist(),
        "feature_weights": {k: float(v) for k, v in DEFAULT_FEATURE_WEIGHTS.items()},
    }
    with open(args.output.with_suffix(".scaler.json"), "w") as f:
        json.dump(scaler_meta, f, indent=2)
    print(f"Wrote processed features to {args.output} (train={len(train)}, val={len(val)}, test={len(test)})")


if __name__ == "__main__":
    main()
