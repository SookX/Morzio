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
        4967.75,
        88.08,
        0.0,
        763.0,
        524.0,
        1278.0,
        3,
        12,
        174.66666666666666,
        106.5,
        259.0,
        78.0,
        81.75709279458403,
        65.20544455795083,
        0.10548034824618792,
        0.08575310754365659,
        0.021438276885914147,
        0.1,
        -640.0,
        5541
    ]

    input_tensor = dataset.prepare_input_list(sample_vector, stats)
    decisions = vae.anomaly_score(input_tensor)
    decision = decisions[0]

    if decision["pass"]:
        print(f"Decision: PASS ({decision['risk']})")
        print(f"Risk epsilon: {decision['epsilon']:.4f}")
    else:
        print(f"Decision: BLOCK ({decision['risk']})")
    print(f"Raw score: {decision['score']:.4f}")
