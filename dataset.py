import torch
from torch.utils.data import Dataset
import os
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, annotations_file, data_dir, transform=None, target_transform=None) -> None:
        self.labels = pd.read_csv(annotations_file)
        self.data_dir = data_dir

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        data_path = os.path.join(self.data_dir, self.labels.iloc[index, 0])
        embedding = torch.load(data_path)
        label = self.labels.iloc[index, 1]
        if label == 2:
            label = 1
        print(self.labels.iloc[index, 0], label)
        return embedding, label
    
# def my_collate(batch):
#     # get max embedding length
#     max
#     # pad all tensors to have [1, max_embedding_length, 1280] size
#     # stack
#     # return
#     # pass this function to collate_fn argument in __init__
#     return None