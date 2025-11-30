from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

EPS = 1e-9
DATA_DIR = Path(__file__).resolve().parent.parent / "vae_annomaly_detection" / "dataset" / "dist"


def parse_currency_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def safe_div(num: float, denom: float) -> float:
    return float(num) / float(denom if abs(denom) > EPS else EPS)


def load_legit_tx_ids(labels_path: Path) -> set[int]:
    with labels_path.open() as f:
        data = json.load(f)
    target = data.get("target", data)
    return {int(k) for k, v in target.items() if str(v).lower() == "no"}


def aggregate_client_expenses(
    tx_path: Path,
    legit_ids: set[int],
    chunksize: int,
    client_ids: set[int],
) -> Dict[int, Dict[str, list]]:
    cols = ["id", "date", "client_id", "amount", "mcc"]
    dtypes = {
        "id": "int64",
        "client_id": "int32",
        "amount": "string",
        "date": "string",
        "mcc": "Int64",
    }

    series: Dict[int, Dict[str, list]] = {cid: {"dates": [], "amounts": [], "mcc": []} for cid in client_ids}

    for chunk in pd.read_csv(tx_path, usecols=cols, dtype=dtypes, chunksize=chunksize):
        chunk = chunk[chunk["client_id"].isin(client_ids)]
        if chunk.empty:
            continue
        chunk = chunk[chunk["id"].isin(legit_ids)]
        if chunk.empty:
            continue

        chunk["date"] = pd.to_datetime(chunk["date"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        chunk["amount_value"] = parse_currency_series(chunk["amount"])

        for cid, grp in chunk.groupby("client_id"):
            rec = series.get(int(cid))
            if rec is None:
                continue
            rec["dates"].extend(grp["date"].tolist())
            rec["amounts"].extend(grp["amount_value"].tolist())
            rec["mcc"].extend(grp["mcc"].tolist())

    return series


def summarize_features_sliding(
    client_ids: Iterable[int],
    series: Dict[int, Dict[str, list]],
    user_df: pd.DataFrame,
    window_days: int = 90,
    stride_days: int = 91,
) -> List[Dict[str, any]]:
    user_df = user_df.set_index("id")
    rows: List[Dict[str, any]] = []
    window_delta = pd.Timedelta(days=window_days)

    for cid in client_ids:
        rec = series.get(cid)
        if rec is None or not rec["dates"]:
            continue

        df_rec = pd.DataFrame(
            {
                "date": pd.to_datetime(rec["dates"]),
                "amount": pd.Series(rec["amounts"], dtype="float"),
                "mcc": pd.Series(rec["mcc"]),
            }
        ).dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        if df_rec.empty:
            continue
        dates = df_rec["date"]
        amounts = df_rec["amount"]

        earliest = dates.min()
        latest = dates.max()
        if pd.isna(earliest) or pd.isna(latest):
            continue

        user_row = user_df.loc[cid] if cid in user_df.index else None
        yearly_income_value = (
            parse_currency_series(pd.Series([user_row["yearly_income"]])).iloc[0] if user_row is not None else np.nan
        )
        estimated_monthly_income = yearly_income_value / 12 if pd.notna(yearly_income_value) else np.nan
        credit_score = float(user_row["credit_score"]) if user_row is not None else np.nan

        snap = earliest + window_delta
        while snap <= latest:
            window_mask = (dates > snap - window_delta) & (dates <= snap)
            window_df = df_rec[window_mask]

            inflow_mask = (dates <= snap) & (amounts > 0)
            if inflow_mask.any():
                inflow_dates = dates[inflow_mask]
                last_inflow_idx = inflow_dates.idxmax()
                last_inflow_amount = float(amounts.loc[last_inflow_idx])
                days_since_last_inflow = float((snap - inflow_dates.max()).days)
            else:
                last_inflow_amount = None
                days_since_last_inflow = None

            expense_df = window_df[window_df["amount"] < 0]
            if expense_df.empty:
                snap += pd.Timedelta(days=stride_days)
                continue
            exp_dates = expense_df["date"]
            exp_abs = expense_df["amount"].abs()

            last_expense = expense_df.iloc[-1]
            current_txn_amount = float(last_expense["amount"])
            current_txn_mcc = int(last_expense["mcc"]) if pd.notna(last_expense["mcc"]) else 0

            last30_mask = exp_dates > snap - pd.Timedelta(days=30)

            total_spend_30d = float(exp_abs[last30_mask].sum()) if exp_abs.size else 0.0
            total_spend_90d = float(exp_abs.sum())
            transaction_count_30d = int(last30_mask.sum())
            transaction_count_90d = int(len(exp_abs))
            avg_txn_amount_30d = float(exp_abs[last30_mask].mean()) if transaction_count_30d else 0.0
            avg_txn_amount_90d = float(exp_abs.mean())
            max_txn_amount_90d = float(exp_abs.max())
            txn_amount_median_90d = float(exp_abs.median())
            spend_volatility_30d = float(exp_abs[last30_mask].std(ddof=0)) if transaction_count_30d else 0.0
            spend_volatility_90d = float(exp_abs.std(ddof=0))

            spend_to_income_ratio_30d = safe_div(total_spend_30d, estimated_monthly_income)
            spend_to_income_ratio_90d = safe_div(total_spend_90d, estimated_monthly_income * 3 if pd.notna(estimated_monthly_income) else 0.0)
            avg_txn_over_income_ratio_90d = safe_div(avg_txn_amount_90d, estimated_monthly_income)
            txn_count_30d_norm = safe_div(transaction_count_30d, 30.0)

            rows.append(
                {
                    "client_id": cid,
                    "snapshot_date": snap.date().isoformat(),
                    "estimated_monthly_income": float(estimated_monthly_income) if pd.notna(estimated_monthly_income) else None,
                    "last_inflow_amount": last_inflow_amount,
                    "days_since_last_inflow": days_since_last_inflow,
                    "credit_score": float(credit_score) if pd.notna(credit_score) else None,
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
                    "current_txn_amount": current_txn_amount,
                    "current_txn_mcc": current_txn_mcc,
                }
            )

            snap += pd.Timedelta(days=stride_days)

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

    series = aggregate_client_expenses(
        tx_path=tx_path,
        legit_ids=legit_ids,
        chunksize=chunksize,
        client_ids=client_ids,
    )
    rows = summarize_features_sliding(client_ids, series, user_df, window_days=90, stride_days=91)
    df = pd.DataFrame(rows).dropna()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", "-o", type=Path, default=Path(__file__).resolve().parent / "output" / "training_features.csv")
    parser.add_argument("--chunksize", type=int, default=200_000)
    args = parser.parse_args()

    out = build_training_dataset(data_dir=args.data_dir, output_path=args.output, chunksize=args.chunksize)
    print(f"Wrote training dataset to {out}")


if __name__ == "__main__":
    main()
