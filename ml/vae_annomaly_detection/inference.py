import torch
import torch.nn as nn
from pipeline import VAEAnomalyDetectionPipeline
from dataset.dataset import CustomDataset
from utils import read_config


if __name__ == "__main__":
    config = read_config("./config.yaml")
    dataset = CustomDataset("./dataset/dist/training_features.csv")
    
    input_dim = 20
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

    

    x = dataset[0].unsqueeze(0)  # Add batch dimension
    print(x.shape)
    print(f"Annomaly score: ", vae.anomaly_score(x))
    