from __future__ import annotations

import math
from typing import Iterable, Optional

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from dema_revenue.main import calculate_dobule_ema


RHO_0 = 0.45
K_F = 5.0
RHO_1 = 0.5
K_A = 5.0
MAX_INSTALLMENTS = 48
ANOMALY_THRESHOLD = 2.0
ANOMALY_K = 1.2
DECLINE_THRESHOLD = 0.90


def sigmoid(x: float) -> float:
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))


def normalize_anomaly_score(
    score: float,
    threshold: float = ANOMALY_THRESHOLD,
    k: float = ANOMALY_K,
) -> float:
    return sigmoid(k * (score - threshold))


def affordability_ratio(transaction_amount: float, monthly_income: float) -> float:
    return abs(transaction_amount) / (monthly_income + 1.0)


def affordability_factor(rho: float, rho_0: float = RHO_0, k_f: float = K_F) -> float:
    return 1.0 - sigmoid(k_f * (rho - rho_0))


def anomaly_weight(rho: float, rho_1: float = RHO_1, k_a: float = K_A) -> float:
    return sigmoid(k_a * (rho - rho_1))


def anomaly_factor(e_normalized: float, rho: float) -> float:
    w = anomaly_weight(rho)
    return 1.0 - w * (e_normalized ** 2)


def round_to_multiple_of_4(n: float) -> int:
    return int(round(n / 4) * 4)


def max_installments(
    monthly_income: float,
    transaction_amount: float,
    anomaly_score: float,
    max_n: int = MAX_INSTALLMENTS,
) -> int:
    e = normalize_anomaly_score(anomaly_score)
    rho = affordability_ratio(transaction_amount, monthly_income)

    if e > DECLINE_THRESHOLD and rho > 0.2:
        return 0

    f_rho = affordability_factor(rho)
    a_e_rho = anomaly_factor(e, rho)
    n_star = max_n * f_rho * a_e_rho
    n = min(max_n, max(0, n_star))
    n = round_to_multiple_of_4(n)

    if 0 < n < 4:
        n = 4

    return int(n)


def estimate_monthly_income_from_inflows(
    inflows: Iterable[float],
    alpha1: float = 0.5,
    alpha2: float = 0.3,
    window_days: int = 30,
) -> Optional[float]:
    inflow_list = list(inflows)
    if not inflow_list:
        return None
    dema_series = calculate_dobule_ema(inflow_list, alpha1=alpha1, alpha2=alpha2)
    smoothed_daily = float(dema_series[-1])
    return smoothed_daily * window_days
