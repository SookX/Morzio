Feature vector for installment approval (dataset-backed)
========================================================

Constrained to fields present in the current data:
- `users_data.csv` (income, credit score)
- `transactions_data.csv` (amount, MCC, timestamp)
- Fraud labels (`train_fraud_labels.json`) are used to keep only legitimate transactions.

Fields (computed for one `client_id` up to `snapshot_date`)
- client_id, snapshot_date: Identifiers and cutoff.
- estimated_monthly_income: `yearly_income / 12` from users_data.
- last_inflow_amount, days_since_last_inflow: Most recent positive transaction and its recency.
- credit_score: From users_data.
- total_spend_30d / total_spend_90d: Sum of expense magnitudes in the window (negative amounts made positive).
- transaction_count_30d / transaction_count_90d: Expense transaction counts.
- avg_txn_amount_30d / avg_txn_amount_90d: Mean expense size in each window.
- max_txn_amount_90d, txn_amount_median_90d: Expense size context.
- spend_volatility_30d / spend_volatility_90d: Std-dev of expense sizes.
- spend_to_income_ratio_30d / spend_to_income_ratio_90d: Spend vs estimated income (90d uses 3× monthly income).
- avg_txn_over_income_ratio_90d: Typical ticket vs income.
- txn_count_30d_norm: Frequency normalized as count/30.
- current_txn_amount, current_txn_mcc: Provided at decision time.

Computation notes
- Parse currency strings (e.g., `$-77.00`) to floats.
- Keep only legitimate (`label == "No"`) transaction IDs and only expenses (`amount_value < 0`) when aggregating.
- Use safe division (add a small epsilon) for all ratios to avoid divide-by-zero.
- Aggregations are relative to the provided `snapshot_date` (inclusive).
