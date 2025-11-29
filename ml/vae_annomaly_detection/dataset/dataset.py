import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler

class CustomDataset(Dataset):
    def __init__(self, filepath):

        self.df = pd.read_csv(filepath)

        self.df = self.df.drop(columns=['client_id', 'snapshot_date'])

        self.df = self.df.dropna(axis=1)

        self.input_array = self.df.values.astype(float)

        self.scaler = StandardScaler()
        self.input_array = self.scaler.fit_transform(self.input_array)

        self.data = torch.tensor(self.input_array, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == "__main__":
    dataset = CustomDataset("./dist/training_features.csv")
    print(f"Dataset length: {len(dataset)}")

    
    sample = dataset[0]
    print(f"Sample shape: {sample.shape}")
    print(f"Sample data: {sample}")
