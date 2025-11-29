import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


DEFAULT_FEATURE_WEIGHTS = {
    "spend_to_income_ratio_30d": 2.0,
    "spend_to_income_ratio_90d": 2.0,
    "avg_txn_over_income_ratio_90d": 1.5,
    "current_txn_amount": 1.5,
    "current_txn_mcc": 1.2,
}


class CustomDataset(Dataset):
    def __init__(self, filepath, feature_weights=None):
        self.df = pd.read_csv(filepath)
        # Drop identifiers/time columns if present
        self.df = self.df.drop(columns=[c for c in ["client_id", "snapshot_date"] if c in self.df.columns])
        # Drop rows with missing values to avoid NaNs in training
        self.df = self.df.dropna()

        self.feature_weights = feature_weights or DEFAULT_FEATURE_WEIGHTS

        X = self.df.values.astype(float)
        # Manual standardization (no sklearn dependency)
        self.means = X.mean(axis=0)
        self.stds = X.std(axis=0)
        self.stds = np.where(self.stds < 1e-9, 1.0, self.stds)
        X = (X - self.means) / self.stds

        # Apply feature weights
        for idx, col in enumerate(self.df.columns):
            w = self.feature_weights.get(col, 1.0)
            if w != 1.0:
                X[:, idx] *= w

        self.data = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def input_dim(self):
        return self.data.shape[1]

    def scaler_stats(self):
        """Return scaler stats and feature order for inference."""
        return {
            "columns": list(self.df.columns),
            "means": self.means.tolist(),
            "stds": self.stds.tolist(),
            "feature_weights": {k: float(v) for k, v in self.feature_weights.items()},
        }


if __name__ == "__main__":
    dataset = CustomDataset("../dema_revenue/output/training_features.csv")
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample shape: {sample.shape}")
    print(f"Sample data: {sample}")
