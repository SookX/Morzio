from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


EPS = 1e-9
DATA_DIR = Path(__file__).resolve().parent.parent / "vae_annomaly_detection" / "dataset" / "dist"


def parse_currency_series(series: pd.Series) -> pd.Series:
    """Convert currency strings like '$-77.00' to floats; invalid -> NaN."""
    cleaned = series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def load_legit_tx_ids(labels_path: Path) -> set[int]:
    """Return the set of transaction IDs labeled as legitimate ('No')."""
    with labels_path.open() as f:
        data = json.load(f)
    target = data.get("target", data)
    return {int(k) for k, v in target.items() if str(v).lower() == "no"}


def safe_div(num: float, denom: float) -> float:
    return float(num) / float(denom if abs(denom) > EPS else EPS)


def build_feature_vector(
    client_id: int,
    snapshot_date: str,
    current_txn_amount: float,
    current_txn_mcc: Optional[int] = None,
    data_dir: Path = DATA_DIR,
    chunksize: int = 200_000,
) -> Dict[str, Any]:
    snap_dt = pd.to_datetime(snapshot_date)

    # Load datasets
    tx_path = data_dir / "transactions_data.csv"
    user_path = data_dir / "users_data.csv"
    labels_path = data_dir / "train_fraud_labels.json"

    legit_ids = load_legit_tx_ids(labels_path)
    tx = pd.read_csv(tx_path, parse_dates=["date"])
    tx["amount_value"] = parse_currency_series(tx["amount"])

    # Filter to legitimate transactions, client, and cutoff date
    tx = tx[tx["id"].isin(legit_ids)]
    tx = tx[(tx["client_id"] == client_id) & (tx["date"] <= snap_dt)]

    # Split inflows/expenses
    inflows = tx[tx["amount_value"] > 0].copy()
    expenses = tx[tx["amount_value"] < 0].copy()
    expenses["spend"] = expenses["amount_value"].abs()

    # User info
    user_df = pd.read_csv(user_path)
    user_row = user_df[user_df["id"] == client_id]
    yearly_income_value = parse_currency_series(user_row["yearly_income"]).iloc[0] if not user_row.empty else float("nan")
    estimated_monthly_income = yearly_income_value / 12 if pd.notna(yearly_income_value) else float("nan")
    credit_score = float(user_row["credit_score"].iloc[0]) if not user_row.empty else float("nan")

    last_inflow_amount = None
    last_inflow_date: Optional[pd.Timestamp] = None
    expense_records: list[tuple[pd.Timestamp, float]] = []

    cols = ["id", "date", "client_id", "amount", "mcc"]
    dtypes = {
        "id": "int64",
        "client_id": "int32",
        "amount": "string",
        "date": "string",
        "mcc": "Int64",
    }
    for chunk in pd.read_csv(tx_path, usecols=cols, dtype=dtypes, chunksize=chunksize):
        chunk = chunk[chunk["client_id"] == client_id]
        if chunk.empty:
            continue
        chunk = chunk[chunk["id"].isin(legit_ids)]
        if chunk.empty:
            continue
        chunk["date"] = pd.to_datetime(chunk["date"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        chunk = chunk[chunk["date"] <= snap_dt]
        if chunk.empty:
            continue
        chunk["amount_value"] = parse_currency_series(chunk["amount"])
        pos = chunk[chunk["amount_value"] > 0]
        if not pos.empty:
            idx = pos["date"].idxmax()
            cand_date = pos.loc[idx, "date"]
            if last_inflow_date is None or cand_date > last_inflow_date:
                last_inflow_date = cand_date
                last_inflow_amount = float(pos.loc[idx, "amount_value"])
        neg = chunk[chunk["amount_value"] < 0]
        if not neg.empty:
            expense_records.extend(zip(neg["date"], neg["amount_value"].abs()))

    expense_df = pd.DataFrame(expense_records, columns=["date", "spend"])
    days_since_last_inflow = float((snap_dt - last_inflow_date).days) if last_inflow_date is not None else None

    def window(df: pd.DataFrame, days: int) -> pd.DataFrame:
        start = snap_dt - pd.Timedelta(days=days)
        return df[df["date"] > start]

    # Aggregations on expenses
    w30 = window(expense_df, 30) if not expense_df.empty else expense_df
    w90 = window(expense_df, 90) if not expense_df.empty else expense_df

    total_spend_30d = float(w30["spend"].sum()) if not w30.empty else 0.0
    total_spend_90d = float(w90["spend"].sum()) if not w90.empty else 0.0
    transaction_count_30d = int(len(w30))
    transaction_count_90d = int(len(w90))
    avg_txn_amount_30d = float(w30["spend"].mean()) if not w30.empty else 0.0
    avg_txn_amount_90d = float(w90["spend"].mean()) if not w90.empty else 0.0
    max_txn_amount_90d = float(w90["spend"].max()) if not w90.empty else 0.0
    txn_amount_median_90d = float(w90["spend"].median()) if not w90.empty else 0.0
    spend_volatility_30d = float(w30["spend"].std(ddof=0)) if not w30.empty else 0.0
    spend_volatility_90d = float(w90["spend"].std(ddof=0)) if not w90.empty else 0.0

    spend_to_income_ratio_30d = safe_div(total_spend_30d, estimated_monthly_income)
    spend_to_income_ratio_90d = safe_div(total_spend_90d, estimated_monthly_income * 3 if pd.notna(estimated_monthly_income) else 0.0)
    avg_txn_over_income_ratio_90d = safe_div(avg_txn_amount_90d, estimated_monthly_income)
    txn_count_30d_norm = safe_div(transaction_count_30d, 30.0)

    feature_vector: Dict[str, Any] = {
        "client_id": client_id,
        "snapshot_date": snap_dt.date().isoformat(),
        "estimated_monthly_income": float(estimated_monthly_income) if pd.notna(estimated_monthly_income) else None,
        "last_inflow_amount": last_inflow_amount,
        "days_since_last_inflow": days_since_last_inflow,
        "credit_score": credit_score if pd.notna(credit_score) else None,
        "total_spend_30d": total_spend_30d,
        "total_spend_90d": total_spend_90d,
        "transaction_count_30d": transaction_count_30d,
        "transaction_count_90d": transaction_count_90d,
        "avg_txn_amount_30d": avg_txn_amount_30d,
        "avg_txn_amount_90d": avg_txn_amount_90d,
        "max_txn_amount_90d": max_txn_amount_90d,
        "txn_amount_median_90d": txn_amount_median_90d,
        "spend_volatility_30d": spend_volatility_30d,
        "spend_volatility_90d": spend_volatility_90d,
        "spend_to_income_ratio_30d": spend_to_income_ratio_30d,
        "spend_to_income_ratio_90d": spend_to_income_ratio_90d,
        "avg_txn_over_income_ratio_90d": avg_txn_over_income_ratio_90d,
        "txn_count_30d_norm": txn_count_30d_norm,
        "current_txn_amount": float(current_txn_amount),
        "current_txn_mcc": int(current_txn_mcc) if current_txn_mcc is not None else None,
    }
    return feature_vector


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a single-client feature vector from legitimate expense transactions.")
    parser.add_argument("--client-id", type=int, required=True, help="Client ID to build features for.")
    parser.add_argument("--snapshot-date", type=str, required=True, help="ISO date (YYYY-MM-DD) cutoff (inclusive).")
    parser.add_argument("--current-txn-amount", type=float, required=True, help="Current transaction amount (negative for expense).")
    parser.add_argument("--current-txn-mcc", type=int, default=None, help="Optional MCC for the current transaction.")
    parser.add_argument("--chunksize", type=int, default=200_000, help="CSV chunk size for streaming.")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Where to write the feature vector JSON. Defaults to ml/dema_revenue/output/features_<client>.json.",
    )
    args = parser.parse_args()

    features = build_feature_vector(
        client_id=args.client_id,
        snapshot_date=args.snapshot_date,
        current_txn_amount=args.current_txn_amount,
        current_txn_mcc=args.current_txn_mcc,
        chunksize=args.chunksize,
    )

    output_path = (
        args.output
        if args.output is not None
        else Path(__file__).resolve().parent / "output" / f"features_client_{args.client_id}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(features, f, indent=2)
    print(f"Wrote feature vector to {output_path}")


if __name__ == "__main__":
    main()
