import torch
from pipeline import VAEAnomalyDetectionPipeline
from dataset.dataset import TransactionDataset
from utils import read_config

# Input vectors from user (credit_score=700 as default)
input_vectors = [
    {
        "estimated_monthly_income": 1900.0,
        "last_inflow_amount": 900.0,
        "days_since_last_inflow": 33,
        "credit_score": 700,
        "total_spend_30d": 510.0,
        "total_spend_90d": 1420.0,
        "transaction_count_30d": 16,
        "transaction_count_90d": 49,
        "avg_txn_amount_30d": 31.87,
        "avg_txn_amount_90d": 28.98,
        "max_txn_amount_90d": 210.0,
        "txn_amount_median_90d": 20.0,
        "spend_volatility_30d": 22.8,
        "spend_volatility_90d": 34.0,
        "spend_to_income_ratio_30d": 0.268,
        "spend_to_income_ratio_90d": 0.249,
        "avg_txn_over_income_ratio_90d": 0.015,
        "txn_count_30d_norm": 0.53,
        "current_txn_amount": -42.00,
        "current_txn_mcc": 5499
    },
    {
        "estimated_monthly_income": 2800.0,
        "last_inflow_amount": 1400.0,
        "days_since_last_inflow": 15,
        "credit_score": 700,
        "total_spend_30d": 410.0,
        "total_spend_90d": 1520.0,
        "transaction_count_30d": 19,
        "transaction_count_90d": 60,
        "avg_txn_amount_30d": 21.57,
        "avg_txn_amount_90d": 25.33,
        "max_txn_amount_90d": 180.0,
        "txn_amount_median_90d": 18.0,
        "spend_volatility_30d": 16.8,
        "spend_volatility_90d": 30.1,
        "spend_to_income_ratio_30d": 0.146,
        "spend_to_income_ratio_90d": 0.181,
        "avg_txn_over_income_ratio_90d": 0.0090,
        "txn_count_30d_norm": 0.63,
        "current_txn_amount": -95.20,
        "current_txn_mcc": 5921
    },
    {
        "estimated_monthly_income": 1200.0,
        "last_inflow_amount": 610.0,
        "days_since_last_inflow": 27,
        "credit_score": 700,
        "total_spend_30d": 620.3,
        "total_spend_90d": 1830.0,
        "transaction_count_30d": 14,
        "transaction_count_90d": 47,
        "avg_txn_amount_30d": 44.3,
        "avg_txn_amount_90d": 38.9,
        "max_txn_amount_90d": 320.0,
        "txn_amount_median_90d": 25.0,
        "spend_volatility_30d": 29.1,
        "spend_volatility_90d": 48.7,
        "spend_to_income_ratio_30d": 0.517,
        "spend_to_income_ratio_90d": 0.508,
        "avg_txn_over_income_ratio_90d": 0.032,
        "txn_count_30d_norm": 0.47,
        "current_txn_amount": -260.00,
        "current_txn_mcc": 5732
    },
    {
        "estimated_monthly_income": 5400.0,
        "last_inflow_amount": 2700.0,
        "days_since_last_inflow": 10,
        "credit_score": 700,
        "total_spend_30d": 900.0,
        "total_spend_90d": 2880.0,
        "transaction_count_30d": 18,
        "transaction_count_90d": 55,
        "avg_txn_amount_30d": 50.0,
        "avg_txn_amount_90d": 52.4,
        "max_txn_amount_90d": 310.0,
        "txn_amount_median_90d": 41.0,
        "spend_volatility_30d": 11.2,
        "spend_volatility_90d": 22.5,
        "spend_to_income_ratio_30d": 0.166,
        "spend_to_income_ratio_90d": 0.177,
        "avg_txn_over_income_ratio_90d": 0.0097,
        "txn_count_30d_norm": 0.60,
        "current_txn_amount": -7.90,
        "current_txn_mcc": 5812
    },
    {
        "estimated_monthly_income": 3100.0,
        "last_inflow_amount": 1550.0,
        "days_since_last_inflow": 18,
        "credit_score": 700,
        "total_spend_30d": 690.4,
        "total_spend_90d": 2080.2,
        "transaction_count_30d": 22,
        "transaction_count_90d": 65,
        "avg_txn_amount_30d": 31.38,
        "avg_txn_amount_90d": 32.00,
        "max_txn_amount_90d": 220.0,
        "txn_amount_median_90d": 20.0,
        "spend_volatility_30d": 14.8,
        "spend_volatility_90d": 28.9,
        "spend_to_income_ratio_30d": 0.222,
        "spend_to_income_ratio_90d": 0.223,
        "avg_txn_over_income_ratio_90d": 0.0103,
        "txn_count_30d_norm": 0.73,
        "current_txn_amount": -38.50,
        "current_txn_mcc": 5411
    }
]

FEATURE_ORDER = [
    "estimated_monthly_income",
    "last_inflow_amount",
    "days_since_last_inflow",
    "credit_score",
    "total_spend_30d",
    "total_spend_90d",
    "transaction_count_30d",
    "transaction_count_90d",
    "avg_txn_amount_30d",
    "avg_txn_amount_90d",
    "max_txn_amount_90d",
    "txn_amount_median_90d",
    "spend_volatility_30d",
    "spend_volatility_90d",
    "spend_to_income_ratio_30d",
    "spend_to_income_ratio_90d",
    "avg_txn_over_income_ratio_90d",
    "txn_count_30d_norm",
    "current_txn_amount",
    "current_txn_mcc",
]

if __name__ == "__main__":
    config = read_config("./config.yaml")
    dataset = TransactionDataset("../data/training_features.csv")
    stats = dataset.scaler_stats()

    input_dim = dataset.input_dim
    hidden_dim = int(config['model']['hidden_dim'])
    latent_dim = int(config['model']['latent_dim'])

    vae = VAEAnomalyDetectionPipeline(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        config=config
    )
    vae.load_model("./checkpoints/best_model.h5")
    print("Model loaded for inference.\n")

    print("=" * 60)
    for i, vec in enumerate(input_vectors, 1):
        # Convert dict to list in correct order
        sample_list = [vec[key] for key in FEATURE_ORDER]
        
        input_tensor = dataset.prepare_input_list(sample_list, stats)
        anomaly, _, _ = vae.anomaly_score(input_tensor)
        
        print(f"Vector {i}:")
        print(f"  Income: ${vec['estimated_monthly_income']:.0f}/mo")
        print(f"  Spend ratio (30d): {vec['spend_to_income_ratio_30d']*100:.1f}%")
        print(f"  Current txn: ${abs(vec['current_txn_amount']):.2f} (MCC {vec['current_txn_mcc']})")
        print(f"  >> Anomaly Score: {anomaly.item():.2f}")
        print("-" * 60)

