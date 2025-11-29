import yaml
import os
import torch

def read_config(config):
    with open(config, "r") as f:
        return yaml.safe_load(f)

def get_device():
    """Get the available device (GPU if available, MPS if available, else CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
    
def data_split(dataset):
    total_len = len(dataset)
    train_len = int(total_len * 0.8)
    val_len = total_len - train_len

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len])
    return train_dataset, val_dataset