import torch
import torch.nn as nn
from pipeline import VAEAnomalyDetectionPipeline
from dataset.dataset import CustomDataset
from sklearn.preprocessing import StandardScaler
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

    x = [3100.0,1550.0,0.0,763.0,524.0,1278.0,3,12,174.66666666666666,106.5,259.0,78.0,81.75709279458403,65.20544455795083,0.10548034824618792,0.08575310754365659,0.021438276885914147,0.1,-64.0,5541]
    x_scaled = dataset.scaler.transform([x])
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)



    #x = dataset[0].unsqueeze(0)  # Add batch dimension
    print(x_tensor.shape)
    print(f"Annomaly score: ", vae.anomaly_score(x_tensor))
    