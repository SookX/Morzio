import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.vae import VAE
from utils import read_config, data_split
from pipeline import VAEAnomalyDetectionPipeline
from dataset.dataset import CustomDataset

if __name__ == "__main__":
    config = read_config("./config.yaml")
    dataset = CustomDataset("./dataset/dist/training_features.csv")
    train_dataset, test_dataset = data_split(dataset)

    train_dataloader = DataLoader(
        train_dataset,
        int(config["training"]["batch_size"]),
        shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        int(config["training"]["batch_size"]),
        shuffle=False
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=2
    )

    input_dim = 20
    hidden_dim = int(config['model']['hidden_dim'])
    latent_dim = int(config['model']['latent_dim'])

    pipeline = VAEAnomalyDetectionPipeline(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        config=config
    )

    print("VAE Anomaly Detection Pipeline initialized.")

    total_params = sum(p.numel() for p in pipeline.model.parameters())
    print(f"Total model parameters: {total_params}")

    pipeline.train(train_dataloader, test_dataloader)
