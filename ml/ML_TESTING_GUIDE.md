# ML Service Testing Guide

This guide explains how to test the ML service end-to-end with realistic Plaid transaction data.

## Overview

The test script simulates the full ML pipeline:
1. **Plaid Transaction Data** → Sample income/expense transactions
2. **Feature Vector Transformation** → 20 features extracted from transactions
3. **VAE Anomaly Detection** → Scoring based on spending patterns
4. **Installment Calculation** → Final decision on max installments

## Prerequisites

Make sure the ML service dependencies are installed:

```bash
cd ml
pip install -r api/requirements.txt
```

## Step 1: Start the ML Service

In one terminal, start the ML API server:

```bash
cd ml
python -m api.main
```

You should see:
```
ML Pipeline loaded successfully
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## Step 2: Run the Test Script

In another terminal, run the test:

```bash
cd ml
python test_ml_service.py
```

### What You'll See

The test script will:
1. Generate sample Plaid transactions (income + expenses)
2. Send a request to `http://localhost:8000/api/predict`
3. Display the ML service response

The **ML terminal** will show detailed debug output:

```
================================================================================
FEATURE VECTOR TRANSFORMATION
================================================================================
estimated_monthly_income           :       2500.0000
last_inflow_amount                 :       2500.0000
days_since_last_inflow             :          5.0000
credit_score                       :        715.0000
total_spend_30d                    :        243.9000
...
================================================================================

================================================================================
ANOMALY DETECTION RESULT
================================================================================
Anomaly Score:                            1.2345
Risk Level:                             Low Risk
Reconstruction Error:                     0.5678
KL Divergence:                            0.1234
================================================================================

================================================================================
INSTALLMENT CALCULATION
================================================================================
Monthly Income:                     £      2500.00
Transaction Amount:                 £       250.00
Affordability Ratio:                       0.0999
Final Installments:                             36
Decision:                              APPROVED
================================================================================
```

The **test script terminal** will show:

```
================================================================================
ML SERVICE END-TO-END TEST
================================================================================

📊 Generated 18 sample transactions

Transaction Summary:
  Income transactions: 3
  Total income (90 days): £7500.00
  Estimated monthly income: £2500.00
  Expense transactions: 15
  Total expenses (90 days): £2951.40

🛒 Current Purchase:
  Amount: £250.00
  MCC Code: 5411 (Grocery Store)

📤 Sending request to http://localhost:8000/api/predict...
✅ Response status: 200

================================================================================
ML SERVICE RESPONSE
================================================================================

✅ Approved: True
📊 Max Installments: 36

💳 Customer can pay £250.00 in up to 36 installments
   Monthly payment: £6.94
```

## Step 3: Test Multiple Scenarios

Run with the `--multiple` flag to test different purchase amounts:

```bash
python test_ml_service.py --multiple
```

This will test:
- Small purchase (£100)
- Medium purchase (£250)
- Large purchase (£500)
- Very large purchase (£1000)

You'll get a summary table showing how installments vary by amount.

## Understanding the Output

### Feature Vector (20 Features)

| Feature | Description | Source |
|---------|-------------|--------|
| `estimated_monthly_income` | Total income / 3 months | Plaid income transactions |
| `last_inflow_amount` | Most recent income | Latest income transaction |
| `days_since_last_inflow` | Days since last income | Current date - last income date |
| `credit_score` | Default value | 715 (hardcoded) |
| `total_spend_30d` | Last 30 days expenses | Plaid expense transactions |
| `total_spend_90d` | Last 90 days expenses | Plaid expense transactions |
| ... | See `feature_builder.py` | ... |

### Anomaly Score

- **Low (< 2.0)**: Normal spending patterns → More installments
- **Medium (2.0-5.0)**: Slightly unusual → Fewer installments
- **High (> 5.0)**: Very unusual → May be declined

### Installment Formula

The formula considers:
- **Affordability Ratio**: `transaction_amount / monthly_income`
- **Anomaly Score**: From VAE model
- **Output**: 0 (declined) to 48 installments (max)

## Modifying the Test Data

Edit `test_ml_service.py` → `generate_sample_transactions()`:

```python
# Add more income
transactions.append({
    "amount": -3000.0,  # £3000 income
    "date": (today - timedelta(days=5)).isoformat(),
    "category": "INCOME"
})

# Add risky spending
transactions.append({
    "amount": 5000.0,  # Large unusual purchase
    "date": (today - timedelta(days=2)).isoformat(),
    "category": "GENERAL_MERCHANDISE"
})
```

## Troubleshooting

### "Could not connect to ML service"

Make sure the ML API is running on port 8000:
```bash
cd ml && python -m api.main
```

### Model checkpoint not found

Verify the checkpoint exists:
```bash
ls ml/vae_annomaly_detection/checkpoints/best_model.h5
```

If missing, you need to train the model first:
```bash
cd ml
python -m vae_annomaly_detection.run
```

## Next Steps

Once the ML service is working, integrate it into the server:

1. Update `PlaidApiController.java` to call the ML service
2. Replace the mock installment logic with ML predictions
3. Test the full flow: Terminal → Server → ML → Response
