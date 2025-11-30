#!/usr/bin/env python3
"""
Full ML Pipeline End-to-End Tests

Tests three scenarios:
1. Normal User - Expected ~44 installments
2. Suspicious Activity - Expected reduced/declined
3. Too Expensive Purchase - Expected declined (0 installments)
"""

import requests
import json
from datetime import date, timedelta
from typing import Dict, List, Any

ML_API_URL = "http://localhost:8000/api/predict"
HEALTH_URL = "http://localhost:8000/health"


def check_server_health() -> bool:
    """Check if ML server is running and healthy."""
    try:
        response = requests.get(HEALTH_URL, timeout=5)
        result = response.json()
        return result.get("status") == "healthy" and result.get("model_loaded")
    except:
        return False


def generate_normal_user_transactions() -> List[Dict[str, Any]]:
    """
    Generate transactions for a normal, stable user.
    - Monthly income: £3000 (3 salary payments)
    - Regular, predictable spending
    - Good financial habits
    """
    today = date.today()
    transactions = []
    
    # Stable income - £3000/month for 3 months
    for i in range(3):
        transactions.append({
            "amount": -3000.0,  # Salary (negative = income in Plaid)
            "date": (today - timedelta(days=5 + i * 30)).isoformat(),
            "category": "INCOME"
        })
    
    # Regular, moderate grocery spending (weekly)
    grocery_amounts = [65.0, 72.0, 58.0, 80.0, 63.0, 70.0, 55.0, 68.0, 75.0, 62.0, 69.0, 71.0]
    for i, amount in enumerate(grocery_amounts):
        transactions.append({
            "amount": amount,
            "date": (today - timedelta(days=i * 7)).isoformat(),
            "category": "FOOD_AND_DRINK"
        })
    
    # Regular rent payments (monthly)
    for i in range(3):
        transactions.append({
            "amount": 900.0,
            "date": (today - timedelta(days=3 + i * 30)).isoformat(),
            "category": "RENT_AND_UTILITIES"
        })
    
    # Utilities (monthly)
    for i in range(3):
        transactions.append({
            "amount": 120.0,
            "date": (today - timedelta(days=10 + i * 30)).isoformat(),
            "category": "GENERAL_SERVICES"
        })
    
    # Occasional entertainment (small, regular)
    for i in range(6):
        transactions.append({
            "amount": 15.0 + (i % 3) * 5,
            "date": (today - timedelta(days=i * 10)).isoformat(),
            "category": "ENTERTAINMENT"
        })
    
    return transactions


def generate_suspicious_activity_transactions() -> List[Dict[str, Any]]:
    """
    Generate transactions showing suspicious/erratic behavior.
    - Monthly income: £2000
    - Highly volatile spending with unusual large purchases
    - Pattern suggests risky financial behavior
    """
    today = date.today()
    transactions = []
    
    # Normal income - £2000/month
    for i in range(3):
        transactions.append({
            "amount": -2000.0,
            "date": (today - timedelta(days=5 + i * 30)).isoformat(),
            "category": "INCOME"
        })
    
    # Erratic large purchases - multiple expensive items in short period
    large_purchases = [
        (1800.0, 3, "ELECTRONICS"),        # Expensive electronics
        (1500.0, 8, "GENERAL_MERCHANDISE"), # Another big purchase
        (1200.0, 15, "TRAVEL"),             # Travel booking
        (900.0, 22, "SHOPPING"),            # Shopping spree
        (1100.0, 28, "ELECTRONICS"),        # More electronics
        (750.0, 45, "GENERAL_MERCHANDISE"), # More purchases
        (1300.0, 55, "TRAVEL"),             # Another trip
    ]
    
    for amount, days_ago, category in large_purchases:
        transactions.append({
            "amount": amount,
            "date": (today - timedelta(days=days_ago)).isoformat(),
            "category": category
        })
    
    # Some small purchases (irregular)
    for i in range(4):
        transactions.append({
            "amount": 25.0 + i * 15,
            "date": (today - timedelta(days=i * 12)).isoformat(),
            "category": "FOOD_AND_DRINK"
        })
    
    return transactions


def generate_too_expensive_transactions() -> List[Dict[str, Any]]:
    """
    Generate transactions for a user with low income trying to buy expensive item.
    - Monthly income: £500
    - Minimal spending history
    - Purchase: £1500 (300% of monthly income)
    """
    today = date.today()
    transactions = []
    
    # Very low income - £500/month
    for i in range(3):
        transactions.append({
            "amount": -500.0,
            "date": (today - timedelta(days=5 + i * 30)).isoformat(),
            "category": "INCOME"
        })
    
    # Minimal spending - just essentials
    for i in range(6):
        transactions.append({
            "amount": 30.0,
            "date": (today - timedelta(days=i * 10)).isoformat(),
            "category": "FOOD_AND_DRINK"
        })
    
    return transactions


def run_test(test_name: str, transactions: List[Dict], purchase_amount: float, 
             expected_min: int, expected_max: int, mcc: int = 5411) -> Dict:
    """Run a single test scenario against the ML API."""
    
    print(f"\n{'=' * 80}")
    print(f"TEST: {test_name}")
    print(f"{'=' * 80}")
    
    # Calculate income summary
    income_txns = [t for t in transactions if t["amount"] < 0]
    expense_txns = [t for t in transactions if t["amount"] > 0]
    total_income = abs(sum(t["amount"] for t in income_txns))
    total_expenses = sum(t["amount"] for t in expense_txns)
    monthly_income = total_income / 3  # 90 days / 3 months
    affordability = purchase_amount / (monthly_income + 1)
    
    print(f"\nTransaction Summary:")
    print(f"  Income transactions: {len(income_txns)}")
    print(f"  Total income (90d): £{total_income:.2f}")
    print(f"  Est. monthly income: £{monthly_income:.2f}")
    print(f"  Expense transactions: {len(expense_txns)}")
    print(f"  Total expenses (90d): £{total_expenses:.2f}")
    print(f"\nPurchase Details:")
    print(f"  Amount: £{purchase_amount:.2f}")
    print(f"  Affordability ratio: {affordability:.2%}")
    print(f"  Expected installments: {expected_min}-{expected_max}")
    
    payload = {
        "transactions": transactions,
        "transaction_amount": purchase_amount,
        "transaction_mcc": mcc
    }
    
    try:
        response = requests.post(ML_API_URL, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            approved = result["approved"]
            installments = result["max_installments"]
            
            # Check if result is in expected range
            in_range = expected_min <= installments <= expected_max
            status = "PASS" if in_range else "FAIL"
            
            print(f"\nResult:")
            print(f"  Approved: {'Yes' if approved else 'No'}")
            print(f"  Max Installments: {installments}")
            print(f"  Expected Range: {expected_min}-{expected_max}")
            print(f"  Test Status: {status}")
            
            return {
                "test_name": test_name,
                "approved": approved,
                "installments": installments,
                "expected_min": expected_min,
                "expected_max": expected_max,
                "passed": in_range,
                "error": None
            }
        else:
            print(f"\nError: HTTP {response.status_code}")
            print(response.text)
            return {
                "test_name": test_name,
                "approved": None,
                "installments": None,
                "expected_min": expected_min,
                "expected_max": expected_max,
                "passed": False,
                "error": f"HTTP {response.status_code}"
            }
            
    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to ML service")
        print("Please ensure the ML service is running: python -m api.main")
        return {
            "test_name": test_name,
            "approved": None,
            "installments": None,
            "expected_min": expected_min,
            "expected_max": expected_max,
            "passed": False,
            "error": "Connection refused"
        }
    except Exception as e:
        print(f"\nError: {e}")
        return {
            "test_name": test_name,
            "approved": None,
            "installments": None,
            "expected_min": expected_min,
            "expected_max": expected_max,
            "passed": False,
            "error": str(e)
        }


def main():
    print("\n" + "#" * 80)
    print("# ML BACKEND FULL PIPELINE TEST")
    print("#" * 80)
    
    # Check server health
    print("\nChecking ML server health...")
    if not check_server_health():
        print("ERROR: ML server is not running or model not loaded!")
        print("Start it with: cd ml && python -m api.main")
        return
    print("ML server is healthy and model is loaded!")
    
    results = []
    
    # Test 1: Normal User (Expected ~44 installments)
    normal_txns = generate_normal_user_transactions()
    result1 = run_test(
        test_name="Normal User - Stable Income & Spending",
        transactions=normal_txns,
        purchase_amount=200.0,  # Small purchase relative to £3000 income
        expected_min=36,
        expected_max=48
    )
    results.append(result1)
    
    # Test 2: Suspicious Activity (Expected reduced/declined)
    suspicious_txns = generate_suspicious_activity_transactions()
    result2 = run_test(
        test_name="Suspicious Activity - Erratic Spending Pattern",
        transactions=suspicious_txns,
        purchase_amount=500.0,
        expected_min=0,
        expected_max=24  # Should be reduced or declined
    )
    results.append(result2)
    
    # Test 3: Too Expensive Purchase (Expected declined)
    expensive_txns = generate_too_expensive_transactions()
    result3 = run_test(
        test_name="Too Expensive - Purchase > Income",
        transactions=expensive_txns,
        purchase_amount=1500.0,  # 300% of £500 income
        expected_min=0,
        expected_max=0  # Should be declined
    )
    results.append(result3)
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"\n{'Test Name':<45} {'Result':<12} {'Installments':<15} {'Expected':<15} {'Status':<10}")
    print("-" * 97)
    
    passed_count = 0
    for r in results:
        if r["error"]:
            result_str = f"ERROR: {r['error']}"
            installments_str = "N/A"
        else:
            result_str = "APPROVED" if r["approved"] else "DECLINED"
            installments_str = str(r["installments"])
        
        expected_str = f"{r['expected_min']}-{r['expected_max']}"
        status = "PASS" if r["passed"] else "FAIL"
        if r["passed"]:
            passed_count += 1
        
        print(f"{r['test_name']:<45} {result_str:<12} {installments_str:<15} {expected_str:<15} {status:<10}")
    
    print("-" * 97)
    print(f"\nTotal: {passed_count}/{len(results)} tests passed")
    
    if passed_count == len(results):
        print("\nAll tests passed! The ML pipeline is working correctly.")
    else:
        print("\nSome tests failed. Review the detailed output above.")
    
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

