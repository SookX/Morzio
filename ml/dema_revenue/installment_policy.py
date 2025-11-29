from __future__ import annotations

import math
from typing import Iterable, Optional

from ml.dema_revenue.main import calculate_dobule_ema


def estimate_monthly_income_from_inflows(
    inflows: Iterable[float],
    alpha1: float = 0.5,
    alpha2: float = 0.3,
    window_days: int = 30,
) -> Optional[float]:
    """
    Estimate a monthly income run rate from a sequence of daily inflow amounts
    using DEMA smoothing. Returns None if not enough data.
    """
    inflow_list = list(inflows)
    if not inflow_list:
        return None
    dema_series = calculate_dobule_ema(inflow_list, alpha1=alpha1, alpha2=alpha2)
    smoothed_daily = float(dema_series[-1])
    # Convert to monthly run rate (approx 30-day month)
    return smoothed_daily * window_days


def max_installments_weekly(
    mse: float,
    purchase_amount: float,
    estimated_monthly_income: Optional[float],
    mse_cap: float = 0.05,
    max_weeks: int = 48,
    min_weeks: int = 4,
    affordability_fraction_weekly: float = 0.25,
) -> int:
    """
    Determine the maximum number of weekly installments given VAE MSE risk and affordability.

    - mse: reconstruction error; higher = riskier. At mse >= mse_cap → zero multiplier.
    - purchase_amount: total cost of the purchase.
    - estimated_monthly_income: monthly income proxy (e.g., DEMA from inflows or yearly/12).
    - affordability_fraction_weekly: max share of weekly income allowed per installment.
    """
    # Risk multiplier shrinks linearly with MSE up to mse_cap
    risk_multiplier = max(0.0, 1.0 - mse / mse_cap)

    # Weekly income estimate
    weekly_income = None
    if estimated_monthly_income is not None and estimated_monthly_income > 0:
        weekly_income = estimated_monthly_income / 4.345  # weeks per month

    # Baseline weeks from affordability
    if weekly_income and weekly_income > 0:
        max_weekly_payment = weekly_income * affordability_fraction_weekly
        if max_weekly_payment <= 0:
            baseline_weeks = min_weeks
        else:
            baseline_weeks = max(min_weeks, math.ceil(purchase_amount / max_weekly_payment))
    else:
        # No income signal; fall back to a conservative baseline
        baseline_weeks = max(min_weeks, min(max_weeks, math.ceil(purchase_amount / 100.0)))

    baseline_weeks = min(baseline_weeks, max_weeks)

    # Apply risk
    allowed_weeks = max(1, int(math.floor(baseline_weeks * risk_multiplier)))
    return min(allowed_weeks, max_weeks)


if __name__ == "__main__":
    # Example usage
    mse = 0.01
    purchase_amount = 500.0
    inflows_daily = [0, 0, 1500, 0, 0, 0, 1500]  # e.g., two paychecks in a week
    est_monthly = estimate_monthly_income_from_inflows(inflows_daily)
    print("Estimated monthly income (DEMA):", est_monthly)
    print("Max weekly installments:", max_installments_weekly(mse, purchase_amount, est_monthly))
