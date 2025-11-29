import yaml
import os
import torch

def load_env(env_path):
    """Load environment variables from a YAML file."""
    with open(env_path, 'r') as file:
        env_vars = yaml.safe_load(file)
    for key, value in env_vars.items():
        os.environ[key] = str(value)

def get_device():
    """Get the available device (GPU if available, MPS if available, else CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')