#!/usr/bin/env python3
"""
Suspicious activity test - tests edge cases where transactions should be declined.
"""

import requests
import json
from datetime import date, timedelta

ML_API_URL = "http://localhost:8000/api/predict"

def test_suspicious_pattern_1():
    """
    Test Case 1: Very low income with large purchase
    - Low income (£500/month)
    - Attempting to buy £1000 item
    - Affordability ratio = 2.0 (200% of income)
    - Expected: DECLINED
    """
    print("\n" + "=" * 80)
    print("TEST 1: LOW INCOME + LARGE PURCHASE")
    print("=" * 80)
    
    today = date.today()
    transactions = []
    
    # Very low income - £500/month
    for i in range(3):
        transactions.append({
            "amount": -500.0,
            "date": (today - timedelta(days=5 + i*30)).isoformat(),
            "category": "INCOME"
        })
    
    # Some small expenses
    for i in range(5):
        transactions.append({
            "amount": 50.0,
            "date": (today - timedelta(days=i*5)).isoformat(),
            "category": "FOOD_AND_DRINK"
        })
    
    # Trying to buy £1000 item
    payload = {
        "transactions": transactions,
        "transaction_amount": 1000.0,
        "transaction_mcc": 5411
    }
    
    response = requests.post(ML_API_URL, json=payload)
    result = response.json()
    
    print(f"\n💰 Monthly Income: £500")
    print(f"🛒 Purchase Amount: £1000")
    print(f"📊 Affordability Ratio: {1000/500:.2f} (200%)")
    print(f"\n{'✅ APPROVED' if result['approved'] else '❌ DECLINED'}")
    print(f"Max Installments: {result['max_installments']}")
    
    return result


def test_suspicious_pattern_2():
    """
    Test Case 2: Highly volatile spending with anomalous activity
    - Normal income (£2000/month)
    - BUT: Very erratic spending pattern + large unusual purchases
    - Expected: DECLINED or very few installments
    """
    print("\n" + "=" * 80)
    print("TEST 2: ERRATIC/ANOMALOUS SPENDING PATTERN")
    print("=" * 80)
    
    today = date.today()
    transactions = []
    
    # Normal income
    for i in range(3):
        transactions.append({
            "amount": -2000.0,
            "date": (today - timedelta(days=5 + i*30)).isoformat(),
            "category": "INCOME"
        })
    
    # Highly erratic spending - multiple large purchases
    large_purchases = [
        (1500.0, 2, "GENERAL_MERCHANDISE"),
        (1200.0, 7, "ELECTRONICS"),
        (1800.0, 12, "TRAVEL"),
        (1000.0, 18, "SHOPPING"),
        (900.0, 25, "GENERAL_MERCHANDISE"),
        (1100.0, 35, "ELECTRONICS"),
    ]
    
    for amount, days_ago, category in large_purchases:
        transactions.append({
            "amount": amount,
            "date": (today - timedelta(days=days_ago)).isoformat(),
            "category": category
        })
    
    # A few small purchases
    for i in range(5):
        transactions.append({
            "amount": 30.0,
            "date": (today - timedelta(days=i*2)).isoformat(),
            "category": "FOOD_AND_DRINK"
        })
    
    # Trying to buy another £500 item
    payload = {
        "transactions": transactions,
        "transaction_amount": 500.0,
        "transaction_mcc": 5411
    }
    
    response = requests.post(ML_API_URL, json=payload)
    result = response.json()
    
    print(f"\n💰 Monthly Income: £2000")
    print(f"🛒 Purchase Amount: £500")
    print(f"💸 Already spent: £7500 in 90 days on large purchases")
    print(f"📊 Spending pattern: Highly volatile and unusual")
    print(f"\n{'✅ APPROVED' if result['approved'] else '❌ DECLINED'}")
    print(f"Max Installments: {result['max_installments']}")
    
    return result


def test_suspicious_pattern_3():
    """
    Test Case 3: No income history
    - 90 days of expenses but no income
    - Expected: DECLINED
    """
    print("\n" + "=" * 80)
    print("TEST 3: NO INCOME HISTORY")
    print("=" * 80)
    
    today = date.today()
    transactions = []
    
    # Only expenses, no income
    for i in range(10):
        transactions.append({
            "amount": 100.0,
            "date": (today - timedelta(days=i*5)).isoformat(),
            "category": "GENERAL_SERVICES"
        })
    
    # Trying to buy £300 item
    payload = {
        "transactions": transactions,
        "transaction_amount": 300.0,
        "transaction_mcc": 5411
    }
    
    response = requests.post(ML_API_URL, json=payload)
    result = response.json()
    
    print(f"\n💰 Monthly Income: £0")
    print(f"🛒 Purchase Amount: £300")
    print(f"❓ No income detected in 90 days")
    print(f"\n{'✅ APPROVED' if result['approved'] else '❌ DECLINED'}")
    print(f"Max Installments: {result['max_installments']}")
    
    return result


def test_normal_user():
    """
    Control test: Normal, stable user
    - Expected: APPROVED with high installments
    """
    print("\n" + "=" * 80)
    print("CONTROL TEST: NORMAL STABLE USER")
    print("=" * 80)
    
    today = date.today()
    transactions = []
    
    # Stable income
    for i in range(3):
        transactions.append({
            "amount": -3000.0,
            "date": (today - timedelta(days=5 + i*30)).isoformat(),
            "category": "INCOME"
        })
    
    # Regular, moderate spending
    for i in range(15):
        transactions.append({
            "amount": 50.0 + (i % 3) * 20,  # Vary between £50-£90
            "date": (today - timedelta(days=i*4)).isoformat(),
            "category": "FOOD_AND_DRINK"
        })
    
    # Rent
    for i in range(2):
        transactions.append({
            "amount": 800.0,
            "date": (today - timedelta(days=10 + i*30)).isoformat(),
            "category": "RENT_AND_UTILITIES"
        })
    
    # Trying to buy £200 item
    payload = {
        "transactions": transactions,
        "transaction_amount": 200.0,
        "transaction_mcc": 5411
    }
    
    response = requests.post(ML_API_URL, json=payload)
    result = response.json()
    
    print(f"\n💰 Monthly Income: £3000")
    print(f"🛒 Purchase Amount: £200")
    print(f"📊 Spending pattern: Stable and predictable")
    print(f"\n{'✅ APPROVED' if result['approved'] else '❌ DECLINED'}")
    print(f"Max Installments: {result['max_installments']}")
    
    return result


if __name__ == "__main__":
    print("\n" + "#" * 80)
    print("# SUSPICIOUS ACTIVITY TESTS")
    print("#" * 80)
    
    results = []
    
    # Run all tests
    results.append(("Normal User (Control)", test_normal_user()))
    results.append(("Low Income + Large Purchase", test_suspicious_pattern_1()))
    results.append(("Erratic Spending", test_suspicious_pattern_2()))
    results.append(("No Income History", test_suspicious_pattern_3()))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"{'Test Case':<35} {'Approved':<12} {'Installments':<15}")
    print("-" * 80)
    
    for test_name, result in results:
        status = "✅ Yes" if result['approved'] else "❌ No"
        print(f"{test_name:<35} {status:<12} {result['max_installments']:<15}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS:")
    print("=" * 80)
    print("The ML model should:")
    print("  ✅ APPROVE normal users with high installments (36-44)")
    print("  ❌ DECLINE or reduce installments for:")
    print("     - Users with income << purchase amount")
    print("     - Users with highly volatile/anomalous spending")
    print("     - Users with no income history")
    print("=" * 80)
