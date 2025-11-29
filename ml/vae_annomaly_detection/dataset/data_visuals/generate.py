from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "dist"
OUTPUT_DIR = BASE_DIR


def _parse_currency(series: pd.Series) -> pd.Series:
    """Convert currency strings like '$1,234.50' to float; invalid values become NaN."""
    cleaned = series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def load_transactions(n_rows: int = 750_000) -> pd.DataFrame:
    """Load a manageable sample of transactions to keep memory low."""
    path = DATA_DIR / "transactions_data.csv"
    df = pd.read_csv(
        path,
        nrows=n_rows,
        parse_dates=["date"],
        dtype={
            "id": "int64",
            "client_id": "int32",
            "card_id": "int32",
            "amount": "string",
            "use_chip": "category",
            "merchant_id": "int32",
            "merchant_city": "category",
            "merchant_state": "category",
            "zip": "float64",
            "mcc": "Int64",
            "errors": "string",
        },
    )
    df["amount_value"] = _parse_currency(df["amount"])
    df["hour"] = df["date"].dt.hour
    return df


def load_mcc_map() -> Dict[str, str]:
    path = DATA_DIR / "mcc_codes.json"
    with path.open() as f:
        return json.load(f)


def save_plot(fig: plt.Figure, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / name
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path.relative_to(Path.cwd())}")


def plot_amount_distribution(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    amounts = df["amount_value"].dropna()
    lower, upper = amounts.quantile([0.01, 0.99])
    clipped = amounts.clip(lower=lower, upper=upper)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(clipped, bins=50, color="#3c6e71", ax=ax)
    ax.set_title("Transaction Amount Distribution (1%-99% clipped)")
    ax.set_xlabel("Amount (USD)")
    ax.set_ylabel("Transactions")
    save_plot(fig, "transactions_amount_distribution.png")


def plot_transactions_by_hour(df: pd.DataFrame) -> None:
    counts = df["hour"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=counts.index, y=counts.values, color="#284b63", ax=ax)
    ax.set_title("Transactions by Hour of Day")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Transactions")
    save_plot(fig, "transactions_by_hour.png")


def plot_top_mcc(df: pd.DataFrame, mcc_map: Dict[str, str], top_n: int = 15) -> None:
    counts = (
        df["mcc"]
        .dropna()
        .astype(int)
        .value_counts()
        .head(top_n)
        .rename_axis("mcc")
        .reset_index(name="transactions")
    )
    counts["mcc_desc"] = counts["mcc"].astype(str).map(mcc_map).fillna("Unknown")

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=counts,
        y="mcc_desc",
        x="transactions",
        palette="viridis",
        ax=ax,
    )
    ax.set_title(f"Top {top_n} Merchant Categories")
    ax.set_xlabel("Transactions")
    ax.set_ylabel("MCC Description")
    save_plot(fig, "top_mcc_categories.png")


def plot_card_brand_distribution() -> None:
    path = DATA_DIR / "cards_data.csv"
    cards = pd.read_csv(path, dtype={"card_brand": "category"})
    counts = cards["card_brand"].value_counts().reset_index()
    counts.columns = ["card_brand", "cards"]

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=counts, x="card_brand", y="cards", palette="crest", ax=ax)
    ax.set_title("Card Brand Distribution")
    ax.set_xlabel("Card Brand")
    ax.set_ylabel("Cards")
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment("right")
    save_plot(fig, "card_brand_distribution.png")


def plot_credit_score_by_gender() -> None:
    path = DATA_DIR / "users_data.csv"
    users = pd.read_csv(path, dtype={"gender": "category", "credit_score": "int32"})

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.boxplot(data=users, x="gender", y="credit_score", palette="Set2", ax=ax)
    ax.set_title("Credit Score by Gender")
    ax.set_xlabel("Gender")
    ax.set_ylabel("Credit Score")
    save_plot(fig, "credit_score_by_gender.png")


def main() -> None:
    print("Loading transactions sample...")
    tx = load_transactions()
    mcc_map = load_mcc_map()

    print("Building visualizations...")
    plot_amount_distribution(tx)
    plot_transactions_by_hour(tx)
    plot_top_mcc(tx, mcc_map)
    plot_card_brand_distribution()
    plot_credit_score_by_gender()

    print("Done.")


if __name__ == "__main__":
    main()
