from torch.utils.data import DataLoader
from utils import read_config, data_split
from pipeline import VAEAnomalyDetectionPipeline
from dataset.dataset import TransactionDataset

if __name__ == "__main__":
    config = read_config("./config.yaml")
    dataset = TransactionDataset("../data/training_features.csv")
    train_dataset, test_dataset = data_split(dataset)

    batch_size = int(config["training"]["batch_size"])
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=2)

    input_dim = dataset.input_dim
    hidden_dim = int(config["model"].get("hidden_dim", 128))
    latent_dim = int(config["model"]["latent_dim"])

    pipeline = VAEAnomalyDetectionPipeline(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        config=config,
    )

    print("VAE Anomaly Detection Pipeline initialized.")
    total_params = sum(p.numel() for p in pipeline.model.parameters())
    print(f"Total model parameters: {total_params}")

    pipeline.train(train_dataloader, test_dataloader)
