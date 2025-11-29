from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class FeatureField:
    """
    Defines a single feature expected by the approval model input vector.
    - name: feature key.
    - dtype: expected numeric or categorical encoding (e.g., float, int, onehot, embedding).
    - description: short purpose statement.
    - window: optional lookback window indicator (e.g., 30d, 90d).
    - notes: optional calculation hints (e.g., uses DEMA, safe division).
    """

    name: str
    dtype: str
    description: str
    window: Optional[str] = None
    notes: Optional[str] = None


# Ordered list describing the feature vector.
INPUT_VECTOR_FIELDS: List[FeatureField] = [
    # Core identifiers/meta
    FeatureField("client_id", "int", "Client identifier from dataset."),
    FeatureField("snapshot_date", "date", "Feature computation cutoff date (inclusive)."),

    # Income & inflows (users_data.csv + transactions_data.csv)
    FeatureField("estimated_monthly_income", "float", "Monthly income derived from yearly_income (yearly/12)."),
    FeatureField("last_inflow_amount", "float", "Most recent positive-amount transaction before snapshot."),
    FeatureField("days_since_last_inflow", "float", "Days since the last inflow transaction."),
    FeatureField("credit_score", "float", "Credit score from users_data.csv."),

    # Spending history (transactions_data.csv) – expenses only (negative amounts)
    FeatureField("total_spend_30d", "float", "Sum of expense magnitudes in the last 30 days.", window="30d"),
    FeatureField("total_spend_90d", "float", "Sum of expense magnitudes in the last 90 days.", window="90d"),
    FeatureField("transaction_count_30d", "int", "Count of expense transactions.", window="30d"),
    FeatureField("transaction_count_90d", "int", "Count of expense transactions.", window="90d"),
    FeatureField("avg_txn_amount_30d", "float", "Mean expense size.", window="30d"),
    FeatureField("avg_txn_amount_90d", "float", "Mean expense size.", window="90d"),
    FeatureField("max_txn_amount_90d", "float", "Maximum expense size.", window="90d"),
    FeatureField("txn_amount_median_90d", "float", "Median expense size.", window="90d"),
    FeatureField("spend_volatility_30d", "float", "Std-dev of expense sizes.", window="30d"),
    FeatureField("spend_volatility_90d", "float", "Std-dev of expense sizes.", window="90d"),

    # Ratios/normalizations
    FeatureField("spend_to_income_ratio_30d", "float", "30d spend vs estimated monthly income.", window="30d", notes="Use epsilon-safe division."),
    FeatureField("spend_to_income_ratio_90d", "float", "90d spend vs 3x estimated monthly income.", window="90d", notes="Use epsilon-safe division."),
    FeatureField("avg_txn_over_income_ratio_90d", "float", "Average expense size vs monthly income.", window="90d", notes="Use epsilon-safe division."),
    FeatureField("txn_count_30d_norm", "float", "Transaction frequency normalized (count/30 days).", window="30d"),

    # Current transaction context (provided at decision time)
    FeatureField("current_txn_amount", "float", "Current transaction amount (negative for expense)."),
    FeatureField("current_txn_mcc", "category", "MCC code of current transaction, if available."),
]
