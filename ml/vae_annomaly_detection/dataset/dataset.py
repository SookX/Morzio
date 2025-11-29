"""Dataset + preprocessing utilities for the VAE anomaly detector."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


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

# Columns that contain strictly positive values and benefit from log scaling.
LOG_SCALE_COLUMNS = {
    "estimated_monthly_income",
    "last_inflow_amount",
    "total_spend_30d",
    "total_spend_90d",
    "transaction_count_30d",
    "transaction_count_90d",
    "avg_txn_amount_30d",
    "avg_txn_amount_90d",
    "max_txn_amount_90d",
    "txn_amount_median_90d",
}

DEFAULT_FEATURE_WEIGHTS: Dict[str, float] = {
    "spend_to_income_ratio_30d": 2.0,
    "spend_to_income_ratio_90d": 2.0,
    "avg_txn_over_income_ratio_90d": 1.5,
    "current_txn_amount": 1.5,
    "current_txn_mcc": 1.2,
}


@dataclass
class ScalerStats:
    columns: List[str]
    means: List[float]
    stds: List[float]
    weights: List[float]

    def transform(self, data: Sequence[float]) -> np.ndarray:
        arr = np.asarray(data, dtype=np.float32)
        arr = np.nan_to_num(arr, copy=False)
        arr = (arr - np.asarray(self.means)) / np.asarray(self.stds)
        return arr * np.asarray(self.weights)


def _log_scale(series: pd.Series) -> pd.Series:
    # log1p mitigates heavy tails on strictly positive features
    clipped = series.clip(lower=0)
    return np.log1p(clipped)


def _clip_outliers(data: np.ndarray, lower_q: float, upper_q: float) -> np.ndarray:
    if data.size == 0:
        return data
    lower = np.quantile(data, lower_q, axis=0)
    upper = np.quantile(data, upper_q, axis=0)
    return np.clip(data, lower, upper)


class TransactionDataset(Dataset):
    """Loads the training CSV, applies preprocessing, and exposes scaler metadata."""

    def __init__(
        self,
        csv_path: str | Path,
        feature_columns: Iterable[str] | None = None,
        feature_weights: Dict[str, float] | None = None,
        clip_quantile: float = 0.995,
    ) -> None:
        self.csv_path = Path(csv_path)
        if feature_columns is None:
            feature_columns = DEFAULT_FEATURE_COLUMNS
        self.columns = [c for c in feature_columns]

        df = pd.read_csv(self.csv_path)

        missing = [col for col in self.columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")

        frame = df[self.columns].copy()

        for col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
            if col in LOG_SCALE_COLUMNS:
                frame[col] = _log_scale(frame[col])

        frame = frame.replace([np.inf, -np.inf], np.nan).dropna(how="any")

        arr = frame.to_numpy(dtype=np.float32)
        if clip_quantile:
            arr = _clip_outliers(arr, 1.0 - clip_quantile, clip_quantile)

        self.means = arr.mean(axis=0)
        self.stds = arr.std(axis=0)
        self.stds = np.where(self.stds < 1e-6, 1.0, self.stds)

        weights = feature_weights or DEFAULT_FEATURE_WEIGHTS
        self.weight_vec = np.array([weights.get(col, 1.0) for col in self.columns], dtype=np.float32)

        arr = (arr - self.means) / self.stds
        arr *= self.weight_vec

        self.data = torch.tensor(arr, dtype=torch.float32)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:  # type: ignore[override]
        return self.data[idx]

    @property
    def input_dim(self) -> int:
        return self.data.shape[1]

    def scaler_stats(self) -> ScalerStats:
        return ScalerStats(
            columns=list(self.columns),
            means=self.means.tolist(),
            stds=self.stds.tolist(),
            weights=self.weight_vec.tolist(),
        )

    def transform_raw(self, values: Sequence[float]) -> torch.Tensor:
        stats = self.scaler_stats()
        scaled = stats.transform(values)
        return torch.tensor(scaled, dtype=torch.float32)
    
    def prepare_input_list(self, values: list[float], scaler: ScalerStats) -> torch.Tensor:
        """
        Takes a raw feature list (already in correct column order),
        applies log-scaling, standardization, and weighting,
        and returns a model-ready tensor.
        """
        if len(values) != len(scaler.columns):
            raise ValueError(
                f"Expected {len(scaler.columns)} features, got {len(values)}"
            )

        processed = []
        for val, col in zip(values, scaler.columns):

            try:
                v = float(val)
            except:
                v = 0.0

            if col in LOG_SCALE_COLUMNS:
                v = np.log1p(max(v, 0))

            processed.append(v)

        transformed = scaler.transform(processed)

        return torch.tensor(transformed, dtype=torch.float32).unsqueeze(0)




if __name__ == "__main__":
    dataset = TransactionDataset("../data/training_features.csv")
    print(f"Dataset length: {len(dataset)}")
    print(f"Input dimension: {dataset.input_dim}")
    stats = dataset.scaler_stats()
    print("First column:", stats.columns[0])
