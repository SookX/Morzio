#!/usr/bin/env python3
"""
Test script to verify demo user profiles against the ML model.
"""

import json
import requests
from pathlib import Path

ML_SERVICE_URL = "http://localhost:8000/api/predict"

def load_user_profile(filename: str) -> dict:
    """Load a user profile JSON file."""
    path = Path(__file__).parent / filename
    with open(path) as f:
        return json.load(f)

def convert_to_ml_format(profile: dict) -> list:
    """Convert Plaid sandbox format to ML API format."""
    transactions = profile["override_accounts"][0]["transactions"]
    
    ml_transactions = []
    for txn in transactions:
        # Plaid sandbox: negative = income, positive = expense
        # ML API expects the same convention
        ml_transactions.append({
            "amount": txn["amount"],
            "date": txn["date_transacted"],
            "category": "INCOME" if txn["amount"] < 0 else "EXPENSE"
        })
    
    return ml_transactions

def test_user(name: str, filename: str, purchase_amount: float, expected_behavior: str):
    """Test a user profile against the ML model."""
    print(f"\n{'='*70}")
    print(f"TESTING: {name}")
    print(f"{'='*70}")
    
    profile = load_user_profile(filename)
    transactions = convert_to_ml_format(profile)
    
    # Calculate income summary
    income_txns = [t for t in transactions if t["amount"] < 0]
    expense_txns = [t for t in transactions if t["amount"] > 0]
    total_income = sum(abs(t["amount"]) for t in income_txns)
    total_expenses = sum(t["amount"] for t in expense_txns)
    
    print(f"\n📊 Profile Summary:")
    print(f"   Name: {profile['override_accounts'][0]['identity']['names'][0]}")
    print(f"   Balance: ${profile['override_accounts'][0]['starting_balance']:,.2f}")
    print(f"   Income transactions: {len(income_txns)}")
    print(f"   Total income (90d): ${total_income:,.2f}")
    print(f"   Est. monthly income: ${total_income/3:,.2f}")
    print(f"   Expense transactions: {len(expense_txns)}")
    print(f"   Total expenses (90d): ${total_expenses:,.2f}")
    
    print(f"\n🛒 Purchase Request: ${purchase_amount:.2f}")
    print(f"   Expected: {expected_behavior}")
    
    # Call ML API
    payload = {
        "transactions": transactions,
        "transaction_amount": purchase_amount,
        "transaction_mcc": 5411
    }
    
    try:
        response = requests.post(ML_SERVICE_URL, json=payload, timeout=10)
        result = response.json()
        
        approved = result.get("approved", False)
        installments = result.get("max_installments", 0)
        
        print(f"\n📤 ML Response:")
        print(f"   Approved: {'✅ Yes' if approved else '❌ No'}")
        print(f"   Max Installments: {installments}")
        
        if approved:
            monthly = purchase_amount / installments if installments > 0 else 0
            print(f"   Monthly Payment: ${monthly:.2f}")
        
        return {"approved": approved, "installments": installments}
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None

def main():
    print("="*70)
    print("MORZIO DEMO - USER PROFILE TESTS")
    print("="*70)
    
    # Check if ML service is running
    try:
        health = requests.get("http://localhost:8000/health", timeout=5)
        if health.status_code != 200:
            print("❌ ML service not healthy. Start it first:")
            print("   cd ml && source vae_annomaly_detection/venv/bin/activate && python -m api.main")
            return
    except:
        print("❌ ML service not running. Start it first:")
        print("   cd ml && source vae_annomaly_detection/venv/bin/activate && python -m api.main")
        return
    
    print("\n✅ ML Service is running\n")
    
    results = []
    
    # Test 1: Suspicious user - should DECLINE for >$40
    r1 = test_user(
        name="🚨 SUSPICIOUS USER (Marcus Risky)",
        filename="user_suspicious.json",
        purchase_amount=50.00,
        expected_behavior="DECLINE (high risk pattern)"
    )
    results.append(("Suspicious ($50)", r1))
    
    # Test 2: Perfect user - should get ~40 installments for $300
    r2 = test_user(
        name="⭐ PERFECT USER (Sarah Johnson)",
        filename="user_perfect.json",
        purchase_amount=300.00,
        expected_behavior="APPROVE with ~40 installments"
    )
    results.append(("Perfect ($300)", r2))
    
    # Test 3: Average user - should get ~20 installments for $250
    r3 = test_user(
        name="📊 AVERAGE USER (Mike Thompson)",
        filename="user_average.json",
        purchase_amount=250.00,
        expected_behavior="APPROVE with ~20-24 installments"
    )
    results.append(("Average ($250)", r3))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'User Profile':<25} {'Purchase':<12} {'Result':<12} {'Installments':<12}")
    print("-"*70)
    
    for name, result in results:
        if result:
            status = "APPROVED" if result["approved"] else "DECLINED"
            inst = result["installments"]
            print(f"{name:<25} {'':>12} {status:<12} {inst:<12}")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()

