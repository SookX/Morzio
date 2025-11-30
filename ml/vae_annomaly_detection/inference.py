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
    4967.75,                 # estimated_monthly_income
    88.08,                   # last_inflow_amount
    0.0,                     # days_since_last_inflow
    763.0,                   # credit_score
    524.0,                   # total_spend_30d
    1278.0,                  # total_spend_90d
    3,                       # transaction_count_30d
    12,                      # transaction_count_90d
    174.66666666666666,      # avg_txn_amount_30d
    106.5,                   # avg_txn_amount_90d
    259.0,                   # max_txn_amount_90d
    78.0,                    # txn_amount_median_90d
    81.75709279458403,       # spend_volatility_30d
    65.20544455795083,       # spend_volatility_90d
    0.10548034824618792,     # spend_to_income_ratio_30d
    0.08575310754365659,     # spend_to_income_ratio_90d
    0.021438276885914147,    # avg_txn_over_income_ratio_90d
    0.1,                     # txn_count_30d_norm
    -640.0,                   # current_txn_amount
    5541                     # current_txn_mcc
]

    input_tensor = dataset.prepare_input_list(
        sample_vector,stats
    )

    decisions = vae.anomaly_score(input_tensor)
    decision = decisions[0]
    if decision["pass"]:
        print(f"Decision: PASS ({decision['risk']})")
        print(f"Risk epsilon: {decision['epsilon']:.4f}")
    else:
        print(f"Decision: BLOCK ({decision['risk']})")
    print(f"Raw score: {decision['score']:.4f}")
    
