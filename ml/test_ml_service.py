#!/usr/bin/env python3
"""
Test script for ML service end-to-end testing.
Simulates Plaid transaction data and tests the full ML pipeline.
"""

import requests
import json
from datetime import date, timedelta

# Configuration
ML_API_URL = "http://localhost:8000/api/predict"

def generate_sample_transactions():
    """
    Generate sample transactions that simulate Plaid data.
    Plaid format: positive amounts are expenses, negative amounts are income.
    """
    today = date.today()
    
    transactions = []
    
    # Income transactions (negative amounts in Plaid)
    transactions.append({
        "amount": -2500.0,  # Monthly salary
        "date": (today - timedelta(days=5)).isoformat(),
        "category": "INCOME"
    })
    
    transactions.append({
        "amount": -2500.0,  # Previous month salary
        "date": (today - timedelta(days=35)).isoformat(),
        "category": "INCOME"
    })
    
    transactions.append({
        "amount": -2500.0,  # Two months ago salary
        "date": (today - timedelta(days=65)).isoformat(),
        "category": "INCOME"
    })
    
    # Expense transactions (positive amounts)
    # Groceries
    transactions.append({"amount": 45.50, "date": (today - timedelta(days=1)).isoformat(), "category": "FOOD_AND_DRINK"})
    transactions.append({"amount": 67.20, "date": (today - timedelta(days=4)).isoformat(), "category": "FOOD_AND_DRINK"})
    transactions.append({"amount": 52.30, "date": (today - timedelta(days=7)).isoformat(), "category": "FOOD_AND_DRINK"})
    transactions.append({"amount": 78.90, "date": (today - timedelta(days=10)).isoformat(), "category": "FOOD_AND_DRINK"})
    
    # Utilities
    transactions.append({"amount": 120.00, "date": (today - timedelta(days=3)).isoformat(), "category": "GENERAL_SERVICES"})
    transactions.append({"amount": 85.00, "date": (today - timedelta(days=33)).isoformat(), "category": "GENERAL_SERVICES"})
    
    # Entertainment
    transactions.append({"amount": 35.00, "date": (today - timedelta(days=2)).isoformat(), "category": "ENTERTAINMENT"})
    transactions.append({"amount": 25.00, "date": (today - timedelta(days=8)).isoformat(), "category": "ENTERTAINMENT"})
    
    # Rent
    transactions.append({"amount": 1200.00, "date": (today - timedelta(days=6)).isoformat(), "category": "RENT_AND_UTILITIES"})
    transactions.append({"amount": 1200.00, "date": (today - timedelta(days=36)).isoformat(), "category": "RENT_AND_UTILITIES"})
    
    # Transport
    transactions.append({"amount": 15.00, "date": (today - timedelta(days=1)).isoformat(), "category": "TRANSPORTATION"})
    transactions.append({"amount": 12.50, "date": (today - timedelta(days=5)).isoformat(), "category": "TRANSPORTATION"})
    
    return transactions


def test_ml_service(transaction_amount=250.0, transaction_mcc=5411):
    """
    Test the ML service with sample data.
    
    Args:
        transaction_amount: Amount of the current purchase (in £)
        transaction_mcc: Merchant Category Code (5411 = grocery store)
    """
    print("=" * 80)
    print("ML SERVICE END-TO-END TEST")
    print("=" * 80)
    
    # Generate sample transactions
    transactions = generate_sample_transactions()
    
    print(f"\n📊 Generated {len(transactions)} sample transactions")
    print("\nTransaction Summary:")
    
    income = [t for t in transactions if t["amount"] < 0]
    expenses = [t for t in transactions if t["amount"] > 0]
    
    total_income = abs(sum(t["amount"] for t in income))
    total_expenses = sum(t["amount"] for t in expenses)
    
    print(f"  Income transactions: {len(income)}")
    print(f"  Total income (90 days): £{total_income:.2f}")
    print(f"  Estimated monthly income: £{total_income / 3:.2f}")
    print(f"  Expense transactions: {len(expenses)}")
    print(f"  Total expenses (90 days): £{total_expenses:.2f}")
    
    # Prepare request payload
    payload = {
        "transactions": transactions,
        "transaction_amount": transaction_amount,
        "transaction_mcc": transaction_mcc
    }
    
    print(f"\n🛒 Current Purchase:")
    print(f"  Amount: £{transaction_amount:.2f}")
    print(f"  MCC Code: {transaction_mcc} (Grocery Store)")
    
    print(f"\n📤 Sending request to {ML_API_URL}...")
    
    try:
        response = requests.post(ML_API_URL, json=payload, timeout=10)
        
        print(f"✅ Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            print("\n" + "=" * 80)
            print("ML SERVICE RESPONSE")
            print("=" * 80)
            print(f"\n✅ Approved: {result['approved']}")
            print(f"📊 Max Installments: {result['max_installments']}")
            
            if result['approved']:
                print(f"\n💳 Customer can pay £{transaction_amount:.2f} in up to {result['max_installments']} installments")
                print(f"   Monthly payment: £{transaction_amount / result['max_installments']:.2f}")
            else:
                print("\n❌ Purchase DECLINED - Risk too high")
            
            return result
        else:
            print(f"\n❌ Error: {response.status_code}")
            print(response.text)
            return None
            
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to ML service")
        print("   Please ensure the ML service is running:")
        print("   cd ml && python -m api.main")
        return None
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        return None


def test_multiple_scenarios():
    """Run multiple test scenarios."""
    print("\n" + "=" * 80)
    print("TESTING MULTIPLE SCENARIOS")
    print("=" * 80)
    
    scenarios = [
        {"amount": 100.0, "mcc": 5411, "name": "Small grocery purchase (£100)"},
        {"amount": 250.0, "mcc": 5411, "name": "Medium grocery purchase (£250)"},
        {"amount": 500.0, "mcc": 5411, "name": "Large grocery purchase (£500)"},
        {"amount": 1000.0, "mcc": 5411, "name": "Very large purchase (£1000)"},
    ]
    
    results = []
    for scenario in scenarios:
        print(f"\n{'=' * 80}")
        print(f"Scenario: {scenario['name']}")
        result = test_ml_service(scenario["amount"], scenario["mcc"])
        if result:
            results.append({
                "scenario": scenario["name"],
                "amount": scenario["amount"],
                "approved": result["approved"],
                "installments": result["max_installments"]
            })
    
    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Scenario':<40} {'Amount':>10} {'Approved':>10} {'Installments':>12}")
    print("-" * 80)
    for r in results:
        status = "✅ Yes" if r["approved"] else "❌ No"
        print(f"{r['scenario']:<40} £{r['amount']:>8.2f} {status:>10} {r['installments']:>12}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--multiple":
        test_multiple_scenarios()
    else:
        # Single test with default values
        test_ml_service(transaction_amount=250.0, transaction_mcc=5411)
        
        print("\n" + "=" * 80)
        print("TIP: Run with --multiple flag to test multiple scenarios:")
        print("  python test_ml_service.py --multiple")
        print("=" * 80)
