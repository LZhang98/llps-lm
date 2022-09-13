import torch
from torch.utils.data import Dataset
import os
import pandas as pd

class DeePhaseDataset(Dataset):
    def __init__(self, annotations_file, data_dir, transform=None, target_transform=None) -> None:
        self.labels = pd.read_csv(annotations_file)
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        data_path = os.path.join(self.data_dir, self.labels.iloc[index, 0])
        embedding = torch.load(data_path)
        label = self.labels[index, 1]
        if self.transform:
            embedding = self.transform(embedding)
        if self.target_transform:
            label = self.target_transform(label)
        return embedding, label