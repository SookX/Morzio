import torch
import torch.nn as nn
from torch.nn import functional as F
import os
from utils import get_device
from tqdm import tqdm

import torch
import torch.nn as nn
import os
from utils import get_device
from model.vae import VAE
from tqdm import tqdm


class VAEAnomalyDetectionPipeline:
    def __init__(self, input_dim, hidden_dim, latent_dim, config):
        self.device = get_device()

        self.model = VAE(input_dim, hidden_dim, latent_dim).to(self.device)

        self.config = config
        self.training_cfg = config['training']
        self.logging_cfg = config['logging']
        self.callbacks_cfg = config['callbacks']

        lr = float(self.training_cfg['learning_rate'])
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        self.criterion = nn.MSELoss(reduction="mean")

        self.best_val_loss = float("inf")
        self.early_stop_counter = 0


    def compute_loss(self, x, reconstructed, mu, logvar):
        recon_loss = self.criterion(reconstructed, x)

        kld_loss = -0.5 * torch.mean( # KL Divergence Loss
            1 + logvar - mu.pow(2) - logvar.exp()
        )

        total_loss = recon_loss + kld_loss
        return total_loss, recon_loss, kld_loss

    @torch.inference_mode()
    def evaluate(self, dataloader):
        self.model.eval()

        total_loss = 0.0
        total_recon = 0.0
        total_kld = 0.0

        for batch in dataloader:
            batch = batch.to(self.device)
            reconstructed, mu, logvar = self.model(batch)
            loss, recon_loss, kld_loss = self.compute_loss(batch, reconstructed, mu, logvar)

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kld += kld_loss.item()

        n = len(dataloader)
        return (
            total_loss / n,
            total_recon / n,
            total_kld / n
        )


    def train(self, train_dataloader, val_dataloader=None):

        epochs = self.training_cfg['epochs']
        log_freq = self.logging_cfg["log_frequency"]

        save_best = self.callbacks_cfg["model_checkpoint"]["enabled"]
        checkpoint_path = self.callbacks_cfg["model_checkpoint"]["filepath"]

        early_stop_enabled = self.callbacks_cfg["early_stopping"]["enabled"]
        patience = self.callbacks_cfg["early_stopping"]["patience"]

        for epoch in range(1, epochs + 1):

            self.model.train()
            total_loss = 0.0

            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}", leave=False):
                batch = batch.to(self.device)

                self.optimizer.zero_grad()
                reconstructed, mu, logvar = self.model(batch)

                loss, _, _ = self.compute_loss(batch, reconstructed, mu, logvar)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_dataloader)

            if val_dataloader is not None:
                val_loss, recon_loss, kld_loss = self.evaluate(val_dataloader)
            else:
                val_loss = recon_loss = kld_loss = None

            if epoch % log_freq == 0:
                print(f"\nEpoch {epoch}/{epochs}:")
                print(f"  Train Loss: {avg_train_loss:.4f}")
                if val_loss is not None:
                    print(f"  Val Loss:   {val_loss:.4f} (Recon: {recon_loss:.4f}, KL: {kld_loss:.4f})")


            if early_stop_enabled and val_loss is not None:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.early_stop_counter = 0

                    if save_best:
                        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                        torch.save(self.model.state_dict(), checkpoint_path)
                else:
                    self.early_stop_counter += 1
                    if self.early_stop_counter >= patience:
                        print(f"Early stopping triggered at epoch {epoch}")
                        break

        print("Training complete.")

    def load_model(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded from {checkpoint_path}")
    
    def inference(self, x):
        self.model.eval()
        x = x.to(self.device)
        with torch.inference_mode():
            reconstructed, mu, logvar = self.model(x)
            #loss, recon_loss, kld_loss = self.compute_loss(x, reconstructed, mu, logvar)
        return reconstructed, mu, logvar 
    
    def anomaly_score(self, vector, alpha=1.0, beta=0.1):

        vector = vector.to(self.device)

        reconstructed, mu, logvar = self.inference(vector)

    
        recon_error = F.mse_loss(reconstructed, vector, reduction='none')
        recon_error = recon_error.mean(dim=-1)

        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

        anomaly = alpha * recon_error + beta * kl

        return (
            anomaly.detach().cpu(),
            recon_error.detach().cpu(),
            kl.detach().cpu()
        )