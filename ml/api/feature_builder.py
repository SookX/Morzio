from datetime import date, timedelta
from typing import List, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class Transaction:
    amount: float
    date: date
    category: Optional[str] = None


DEFAULT_CREDIT_SCORE = 715.0
EPS = 1e-9


def safe_div(num: float, denom: float) -> float:
    return float(num) / float(denom if abs(denom) > EPS else EPS)


def build_feature_vector(
    transactions: List[Transaction],
    current_txn_amount: float,
    current_txn_mcc: int,
    snapshot_date: Optional[date] = None,
) -> List[float]:
    if snapshot_date is None:
        snapshot_date = date.today()

    expenses = []
    incomes = []

    for txn in transactions:
        if txn.date > snapshot_date:
            continue
        if txn.amount > 0:
            expenses.append((txn.date, txn.amount))
        else:
            incomes.append((txn.date, abs(txn.amount)))

    # Income calculations
    total_income_90d = sum(amt for d, amt in incomes if d > snapshot_date - timedelta(days=90))
    estimated_monthly_income = total_income_90d / 3.0 if total_income_90d > 0 else 0.0

    last_inflow_amount = 0.0
    days_since_last_inflow = 90.0
    if incomes:
        incomes_sorted = sorted(incomes, key=lambda x: x[0], reverse=True)
        last_inflow_amount = incomes_sorted[0][1]
        days_since_last_inflow = (snapshot_date - incomes_sorted[0][0]).days

    # Expense calculations
    exp_30d = [(d, a) for d, a in expenses if d > snapshot_date - timedelta(days=30)]
    exp_90d = [(d, a) for d, a in expenses if d > snapshot_date - timedelta(days=90)]

    amounts_30d = [a for _, a in exp_30d]
    amounts_90d = [a for _, a in exp_90d]

    total_spend_30d = sum(amounts_30d)
    total_spend_90d = sum(amounts_90d)
    transaction_count_30d = len(amounts_30d)
    transaction_count_90d = len(amounts_90d)

    avg_txn_amount_30d = np.mean(amounts_30d) if amounts_30d else 0.0
    avg_txn_amount_90d = np.mean(amounts_90d) if amounts_90d else 0.0
    max_txn_amount_90d = max(amounts_90d) if amounts_90d else 0.0
    txn_amount_median_90d = np.median(amounts_90d) if amounts_90d else 0.0

    spend_volatility_30d = np.std(amounts_30d, ddof=0) if amounts_30d else 0.0
    spend_volatility_90d = np.std(amounts_90d, ddof=0) if amounts_90d else 0.0

    spend_to_income_ratio_30d = safe_div(total_spend_30d, estimated_monthly_income)
    spend_to_income_ratio_90d = safe_div(total_spend_90d, estimated_monthly_income * 3)
    avg_txn_over_income_ratio_90d = safe_div(avg_txn_amount_90d, estimated_monthly_income)
    txn_count_30d_norm = safe_div(transaction_count_30d, 30.0)

    feature_vector = [
        estimated_monthly_income,
        last_inflow_amount,
        days_since_last_inflow,
        DEFAULT_CREDIT_SCORE,
        total_spend_30d,
        total_spend_90d,
        transaction_count_30d,
        transaction_count_90d,
        avg_txn_amount_30d,
        avg_txn_amount_90d,
        max_txn_amount_90d,
        txn_amount_median_90d,
        spend_volatility_30d,
        spend_volatility_90d,
        spend_to_income_ratio_30d,
        spend_to_income_ratio_90d,
        avg_txn_over_income_ratio_90d,
        txn_count_30d_norm,
        -abs(current_txn_amount),
        current_txn_mcc,
    ]

    return feature_vector

