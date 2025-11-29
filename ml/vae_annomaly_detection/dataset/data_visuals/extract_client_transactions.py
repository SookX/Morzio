from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_currency(series: pd.Series) -> pd.Series:
    """Convert currency strings like '$1,234.56' to floats; invalid -> NaN."""
    cleaned = series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def extract_client(client_id: int, data_dir: Path, output: Path, chunksize: int = 200_000) -> int:
    tx_path = data_dir / "transactions_data.csv"
    if not tx_path.exists():
        raise FileNotFoundError(f"Missing transactions file at {tx_path}")

    frames = []
    for chunk in pd.read_csv(
        tx_path,
        chunksize=chunksize,
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
    ):
        match = chunk[chunk["client_id"] == client_id]
        if not match.empty:
            frames.append(match.copy())

    if not frames:
        raise ValueError(f"No transactions found for client_id {client_id}")

    result = pd.concat(frames, ignore_index=True)
    result["amount_value"] = parse_currency(result["amount"])
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return len(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract all transactions for a client into a CSV table.")
    parser.add_argument("--client-id", "-c", type=int, default=1223, help="Client ID to extract.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "dist",
        help="Path to dataset dist folder containing transactions_data.csv.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output CSV path. Defaults to data_visuals/client_<id>_transactions.csv",
    )
    parser.add_argument("--chunksize", type=int, default=200_000, help="CSV read chunk size.")
    args = parser.parse_args()

    data_dir = args.dataset_dir.resolve()
    output = (
        args.output
        if args.output is not None
        else Path(__file__).resolve().parent / f"client_{args.client_id}_transactions.csv"
    )

    count = extract_client(args.client_id, data_dir=data_dir, output=output, chunksize=args.chunksize)
    print(f"Wrote {count} rows to {output}")


if __name__ == "__main__":
    main()
