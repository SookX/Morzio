import torch
from pipeline import VAEAnomalyDetectionPipeline
from dataset.dataset import TransactionDataset
from utils import read_config


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
    print("Model loaded for inference.")

    sample_vector = [
    1500.0,      # estimated_monthly_income (low)
    80.0,        # last_inflow_amount (tiny inflow recently)
    28.0,        # days_since_last_inflow (long gap - instability)
    510.0,       # credit_score (poor)
    2500.0,      # total_spend_30d (very high vs income)
    6200.0,      # total_spend_90d (extremely high)
    45,          # transaction_count_30d (unusually high)
    130,         # transaction_count_90d (unusually high)
    180.0,       # avg_txn_amount_30d (high relative to income)
    165.0,       # avg_txn_amount_90d
    2100.0,      # max_txn_amount_90d (suspiciously large)
    950.0,       # txn_amount_median_90d (too high)
    140.0,       # spend_volatility_30d (unstable + high variance)
    310.0,       # spend_volatility_90d (extreme volatility)
    1.65,        # spend_to_income_ratio_30d (>100% of income = bad)
    2.40,        # spend_to_income_ratio_90d (>200% of income)
    0.32,        # avg_txn_over_income_ratio_90d (abnormally high)
    0.95,        # txn_count_30d_norm (near max risk)
    -1800.0,     # current_txn_amount (very large negative withdrawal)
    7995         # current_txn_mcc (high-risk / unusual merchant)
]



    input_tensor = dataset.prepare_input_list(
        sample_vector,stats
    )

    anomaly, _, _ = vae.anomaly_score(input_tensor)
    print("Anomaly score:", anomaly.item())
    #print("Recon:", recon_error.item(), " KL:", kl.item())
    
