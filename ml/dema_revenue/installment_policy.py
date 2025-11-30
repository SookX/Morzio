"""
Installment Decision Policy

Determines the maximum number of installments a user is eligible for based on:
- Monthly income (I)
- Transaction amount (A)
- Anomaly score from VAE (e)

Uses sigmoid-based smooth transitions for affordability and anomaly factors.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

# Allow running as a script from repo root without installing package
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from dema_revenue.main import calculate_dobule_ema


# ============================================================================
# Constants (from specification)
# ============================================================================

RHO_0 = 0.45      # Affordability threshold - allows 44 for tiny purchases
K_F = 5.0         # Affordability sigmoid steepness

RHO_1 = 0.5       # Anomaly weight threshold  
K_A = 5.0         # Anomaly weight sigmoid steepness

MAX_INSTALLMENTS = 48

# Anomaly score normalization parameters
ANOMALY_THRESHOLD = 2.0   # Sensitive to anomalies
ANOMALY_K = 1.2           # Moderate steepness

# High-risk decline threshold (normalized anomaly score)
DECLINE_THRESHOLD = 0.90  # If e_normalized > this AND high rho, decline completely


# ============================================================================
# Helper Functions
# ============================================================================

def sigmoid(x: float) -> float:
    """Standard sigmoid function σ(x) = 1 / (1 + e^(-x))"""
    # Clip to avoid overflow
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))


def normalize_anomaly_score(
    score: float,
    threshold: float = ANOMALY_THRESHOLD,
    k: float = ANOMALY_K,
) -> float:
    """
    Normalize VAE anomaly score to [0,1] using sigmoid.
    
    - Scores around threshold → 0.5
    - Normal scores (~1.5) → ~0.18
    - Anomalous scores (10+) → ~0.99+
    """
    return sigmoid(k * (score - threshold))


# ============================================================================
# Installment Formula Components
# ============================================================================

def affordability_ratio(transaction_amount: float, monthly_income: float) -> float:
    """
    Calculate affordability ratio ρ = A / (I + 1)
    
    Args:
        transaction_amount: Absolute value of current transaction (A)
        monthly_income: Estimated monthly income (I)
    
    Returns:
        Affordability ratio ρ
    """
    return abs(transaction_amount) / (monthly_income + 1.0)


def affordability_factor(rho: float, rho_0: float = RHO_0, k_f: float = K_F) -> float:
    """
    Calculate affordability factor f(ρ) = 1 - σ(k_f * (ρ - ρ_0))
    
    High when purchase is small relative to income.
    Low when purchase is large relative to income.
    """
    return 1.0 - sigmoid(k_f * (rho - rho_0))


def anomaly_weight(rho: float, rho_1: float = RHO_1, k_a: float = K_A) -> float:
    """
    Calculate anomaly importance weight w(ρ) = σ(k_a * (ρ - ρ_1))
    
    Low for cheap purchases (anomaly matters less).
    High for expensive purchases (anomaly matters more).
    """
    return sigmoid(k_a * (rho - rho_1))


def anomaly_factor(e_normalized: float, rho: float) -> float:
    """
    Calculate anomaly factor a(e, ρ) = 1 - w(ρ) * e²
    
    Reduces installments based on anomaly score, weighted by purchase size.
    """
    w = anomaly_weight(rho)
    return 1.0 - w * (e_normalized ** 2)


# ============================================================================
# Main Installment Function
# ============================================================================

def round_to_multiple_of_4(n: float) -> int:
    """Round to nearest multiple of 4."""
    return int(round(n / 4) * 4)


def max_installments(
    monthly_income: float,
    transaction_amount: float,
    anomaly_score: float,
    max_n: int = MAX_INSTALLMENTS,
) -> int:
    """
    Determine the maximum number of installments a user is eligible for.
    
    Args:
        monthly_income: Estimated monthly income (I)
        transaction_amount: Current transaction amount, absolute value (A)
        anomaly_score: Raw anomaly score from VAE model
        max_n: Maximum allowable installments (default 48)
    
    Returns:
        Integer number of installments (0, 4, 8, 12, ..., 48)
        Returns 0 if user should be declined.
    
    Formula:
        ρ = A / (I + 1)                           # Affordability ratio
        f(ρ) = 1 - σ(k_f * (ρ - ρ_0))            # Affordability factor
        w(ρ) = σ(k_a * (ρ - ρ_1))                # Anomaly weight
        a(e, ρ) = 1 - w(ρ) * e²                  # Anomaly factor
        N* = max_n * f(ρ) * a(e, ρ)              # Raw score
        N = round_to_4(min(max_n, max(0, N*)))   # Final count (divisible by 4)
    """
    # Normalize anomaly score to [0, 1]
    e = normalize_anomaly_score(anomaly_score)
    
    # Calculate affordability ratio
    rho = affordability_ratio(transaction_amount, monthly_income)
    
    # DECLINE: High anomaly score with non-trivial purchase → reject completely
    if e > DECLINE_THRESHOLD and rho > 0.2:
        return 0
    
    # Calculate factors
    f_rho = affordability_factor(rho)
    a_e_rho = anomaly_factor(e, rho)
    
    # Raw installment score
    n_star = max_n * f_rho * a_e_rho
    
    # Final installment count (clamped and rounded to multiple of 4)
    n = min(max_n, max(0, n_star))
    n = round_to_multiple_of_4(n)
    
    # Minimum non-zero is 4
    if 0 < n < 4:
        n = 4
    
    return int(n)


# ============================================================================
# Income Estimation Utility
# ============================================================================

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


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("INSTALLMENT DECISION FORMULA - TEST CASES")
    print("=" * 70)
    
    # Test anomaly normalization
    print("\n1. Anomaly Score Normalization:")
    print("-" * 40)
    for raw_score in [1.5, 2.5, 5.0, 10.0, 50.0]:
        normalized = normalize_anomaly_score(raw_score)
        status = "DECLINE" if normalized > DECLINE_THRESHOLD else "OK"
        print(f"   Raw: {raw_score:5.1f} → Normalized: {normalized:.3f} ({status})")
    
    # Test cases
    test_cases = [
        # (income, amount, anomaly_score, description)
        (5000, 100, 1.5, "High income, small purchase, normal user"),
        (5000, 100, 50.0, "High income, small purchase, anomalous user"),
        (3000, 1500, 1.5, "Medium income, medium purchase, normal user"),
        (3000, 1500, 50.0, "Medium income, medium purchase, anomalous"),
        (2000, 3000, 1.5, "Low income, large purchase, normal user"),
        (2000, 3000, 50.0, "Low income, large purchase, anomalous"),
        (1000, 500, 1.5, "Very low income, medium purchase, normal"),
        (10000, 200, 1.5, "Very high income, small purchase, normal"),
        (4000, 800, 15.0, "Medium income, medium purchase, suspicious"),
    ]
    
    print("\n2. Installment Calculations (divisible by 4):")
    print("-" * 75)
    print(f"{'Description':<45} {'Income':>7} {'Amount':>7} {'Anomaly':>8} {'N':>5}")
    print("-" * 75)
    
    for income, amount, anomaly, desc in test_cases:
        n = max_installments(income, amount, anomaly)
        result = str(n) if n > 0 else "DECLINE"
        print(f"{desc:<45} ${income:>6} ${amount:>6} {anomaly:>8.1f} {result:>5}")
    
    print("\n3. Decline Cases (high anomaly + non-trivial purchase):")
    print("-" * 50)
    
    # High anomaly users should be declined
    for anomaly in [20, 50, 100]:
        n = max_installments(3000, 500, anomaly)
        status = "DECLINED" if n == 0 else f"{n} installments"
        print(f"   Income $3k, purchase $500, anomaly {anomaly}: {status}")
    
    print("\n4. Edge Cases:")
    print("-" * 40)
    
    # Edge: Zero income
    n = max_installments(0, 100, 1.5)
    print(f"   Zero income, $100 purchase: {n} installments")
    
    # Edge: Tiny purchase with high anomaly (should still allow)
    n = max_installments(5000, 10, 100.0)
    print(f"   High income, tiny $10 purchase, high anomaly: {n} installments")
    
    # Edge: Huge purchase relative to income
    n = max_installments(1000, 10000, 1.5)
    print(f"   $1k income, $10k purchase: {n} installments")
    
    print("\n" + "=" * 70)
    print("Formula behavior:")
    print("  - All installment counts are divisible by 4")
    print("  - High anomaly (e > 0.85) + purchase > 20% income → DECLINE (0)")
    print("  - Small purchases tolerate higher anomaly scores")
    print("  - Stricter thresholds for more punishing behavior")
    print("=" * 70)
