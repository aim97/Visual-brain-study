import torch
from torch.utils.data import Dataset

# Splitter class
class Splitter(Dataset):

    def __init__(self, dataset, split_path, split_num=0, split_name="train"):
        # Set EEG dataset
        self.dataset = dataset
        # Load split
        loaded = torch.load(split_path)
        self.split_idx = loaded["splits"][split_num][split_name]
        # Filter data
        self.split_idx = [
            i
            for i in self.split_idx
            if 450 <= self.dataset.data[i]["eeg"].size(1) <= 600
        ]
        # Compute size
        self.size = len(self.split_idx)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Get sample from dataset
        eeg, label = self.dataset[self.split_idx[i]] 
        # Return
        return eeg, label
