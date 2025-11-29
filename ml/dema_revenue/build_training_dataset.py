from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

EPS = 1e-9
DATA_DIR = Path(__file__).resolve().parent.parent / "vae_annomaly_detection" / "dataset" / "dist"


def parse_currency_series(series: pd.Series) -> pd.Series:
    """Convert currency strings like '$-77.00' to floats; invalid -> NaN."""
    cleaned = series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def safe_div(num: float, denom: float) -> float:
    return float(num) / float(denom if abs(denom) > EPS else EPS)


def load_legit_tx_ids(labels_path: Path) -> set[int]:
    """Return the set of transaction IDs labeled as legitimate ('No')."""
    with labels_path.open() as f:
        data = json.load(f)
    target = data.get("target", data)
    return {int(k) for k, v in target.items() if str(v).lower() == "no"}


def find_snapshot_date(tx_path: Path, legit_ids: set[int], chunksize: int) -> pd.Timestamp:
    """Find the max transaction date among legitimate transactions."""
    cols = ["id", "date"]
    dtypes = {"id": "int64", "date": "string"}
    max_date: Optional[pd.Timestamp] = None
    for chunk in pd.read_csv(tx_path, usecols=cols, dtype=dtypes, chunksize=chunksize):
        chunk = chunk[chunk["id"].isin(legit_ids)]
        if chunk.empty:
            continue
        dates = pd.to_datetime(chunk["date"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        if dates.empty:
            continue
        chunk_max = dates.max()
        if pd.isna(chunk_max):
            continue
        if max_date is None or chunk_max > max_date:
            max_date = chunk_max
    if max_date is None:
        raise ValueError("No legitimate transactions found to determine snapshot date.")
    return max_date


def aggregate_client_expenses(
    tx_path: Path,
    legit_ids: set[int],
    snapshot_date: pd.Timestamp,
    client_ids: set[int],
    chunksize: int,
) -> Dict[int, Dict[str, any]]:
    """Stream transactions and collect per-client expense/inflow metrics."""
    cols = ["id", "date", "client_id", "amount", "mcc"]
    dtypes = {
        "id": "int64",
        "client_id": "int32",
        "amount": "string",
        "date": "string",
        "mcc": "Int64",
    }

    cutoff_30 = snapshot_date - pd.Timedelta(days=30)
    cutoff_90 = snapshot_date - pd.Timedelta(days=90)

    aggregates: Dict[int, Dict[str, any]] = {
        cid: {
            "last_inflow_amount": None,
            "last_inflow_date": None,
            "expenses_30": [],
            "expenses_90": [],
            "txn_count_30": 0,
            "txn_count_90": 0,
        }
        for cid in client_ids
    }

    for chunk in pd.read_csv(tx_path, usecols=cols, dtype=dtypes, chunksize=chunksize):
        # Filter early by client and legitimacy
        chunk = chunk[chunk["client_id"].isin(client_ids)]
        if chunk.empty:
            continue
        chunk = chunk[chunk["id"].isin(legit_ids)]
        if chunk.empty:
            continue

        chunk["date"] = pd.to_datetime(chunk["date"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        chunk = chunk[chunk["date"] <= snapshot_date]
        if chunk.empty:
            continue

        chunk["amount_value"] = parse_currency_series(chunk["amount"])

        for cid, grp in chunk.groupby("client_id"):
            agg = aggregates.get(int(cid))
            if agg is None:
                continue
            # Inflows (positive)
            inflows = grp[grp["amount_value"] > 0]
            if not inflows.empty:
                idx = inflows["date"].idxmax()
                inflow_date = inflows.loc[idx, "date"]
                if agg["last_inflow_date"] is None or inflow_date > agg["last_inflow_date"]:
                    agg["last_inflow_date"] = inflow_date
                    agg["last_inflow_amount"] = float(inflows.loc[idx, "amount_value"])

            # Expenses (negative)
            expenses = grp[grp["amount_value"] < 0]
            if expenses.empty:
                continue
            expenses = expenses.assign(spend=expenses["amount_value"].abs())

            exp_90 = expenses[expenses["date"] > cutoff_90]
            if not exp_90.empty:
                agg["expenses_90"].extend(exp_90["spend"].tolist())
                agg["txn_count_90"] += len(exp_90)
            exp_30 = expenses[expenses["date"] > cutoff_30]
            if not exp_30.empty:
                agg["expenses_30"].extend(exp_30["spend"].tolist())
                agg["txn_count_30"] += len(exp_30)

    return aggregates


def summarize_features(
    client_ids: Iterable[int],
    aggregates: Dict[int, Dict[str, any]],
    user_df: pd.DataFrame,
    snapshot_date: pd.Timestamp,
) -> List[Dict[str, any]]:
    """Compute final feature rows for each client."""
    user_df = user_df.set_index("id")
    rows: List[Dict[str, any]] = []
    for cid in client_ids:
        agg = aggregates.get(cid)
        if agg is None:
            continue

        user_row = user_df.loc[cid] if cid in user_df.index else None
        yearly_income_value = (
            parse_currency_series(pd.Series([user_row["yearly_income"]])).iloc[0] if user_row is not None else np.nan
        )
        estimated_monthly_income = yearly_income_value / 12 if pd.notna(yearly_income_value) else np.nan
        credit_score = float(user_row["credit_score"]) if user_row is not None else np.nan

        exp30 = np.array(agg["expenses_30"], dtype=float)
        exp90 = np.array(agg["expenses_90"], dtype=float)

        total_spend_30d = float(exp30.sum()) if exp30.size else 0.0
        total_spend_90d = float(exp90.sum()) if exp90.size else 0.0
        avg_txn_amount_30d = float(exp30.mean()) if exp30.size else 0.0
        avg_txn_amount_90d = float(exp90.mean()) if exp90.size else 0.0
        max_txn_amount_90d = float(exp90.max()) if exp90.size else 0.0
        txn_amount_median_90d = float(np.median(exp90)) if exp90.size else 0.0
        spend_volatility_30d = float(exp30.std(ddof=0)) if exp30.size else 0.0
        spend_volatility_90d = float(exp90.std(ddof=0)) if exp90.size else 0.0

        spend_to_income_ratio_30d = safe_div(total_spend_30d, estimated_monthly_income)
        spend_to_income_ratio_90d = safe_div(total_spend_90d, estimated_monthly_income * 3 if pd.notna(estimated_monthly_income) else 0.0)
        avg_txn_over_income_ratio_90d = safe_div(avg_txn_amount_90d, estimated_monthly_income)
        txn_count_30d_norm = safe_div(agg["txn_count_30"], 30.0)

        days_since_last_inflow = (
            float((snapshot_date - agg["last_inflow_date"]).days) if agg["last_inflow_date"] is not None else None
        )

        rows.append(
            {
                "client_id": cid,
                "snapshot_date": snapshot_date.date().isoformat(),
                "estimated_monthly_income": float(estimated_monthly_income) if pd.notna(estimated_monthly_income) else None,
                "last_inflow_amount": agg["last_inflow_amount"],
                "days_since_last_inflow": days_since_last_inflow,
                "credit_score": float(credit_score) if pd.notna(credit_score) else None,
                "total_spend_30d": total_spend_30d,
                "total_spend_90d": total_spend_90d,
                "transaction_count_30d": agg["txn_count_30"],
                "transaction_count_90d": agg["txn_count_90"],
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
                # Current transaction fields left empty for training dataset
                "current_txn_amount": None,
                "current_txn_mcc": None,
            }
        )
    return rows


def build_training_dataset(
    data_dir: Path,
    output_path: Path,
    chunksize: int,
) -> Path:
    tx_path = data_dir / "transactions_data.csv"
    user_path = data_dir / "users_data.csv"
    labels_path = data_dir / "train_fraud_labels.json"

    legit_ids = load_legit_tx_ids(labels_path)
    user_df = pd.read_csv(user_path)
    client_ids = set(user_df["id"].astype(int).tolist())

    snapshot_date = find_snapshot_date(tx_path, legit_ids, chunksize=chunksize)
    aggregates = aggregate_client_expenses(
        tx_path=tx_path,
        legit_ids=legit_ids,
        snapshot_date=snapshot_date,
        client_ids=client_ids,
        chunksize=chunksize,
    )
    rows = summarize_features(client_ids, aggregates, user_df, snapshot_date)
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a training dataset of feature vectors for all clients.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Directory containing source CSVs.")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "training_features.csv",
        help="Output CSV path.",
    )
    parser.add_argument("--chunksize", type=int, default=200_000, help="CSV chunk size for streaming.")
    args = parser.parse_args()

    out = build_training_dataset(data_dir=args.data_dir, output_path=args.output, chunksize=args.chunksize)
    print(f"Wrote training dataset to {out}")


if __name__ == "__main__":
    main()
