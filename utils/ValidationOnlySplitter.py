import torch
from torch.utils.data import Dataset


# Splitter class
class ValidationOnlySplitter(Dataset):

    def __init__(self, dataset, split_path, split_num=0):
        super().__init__()
        # Set EEG dataset
        self.dataset = dataset
        # Load split
        loaded = torch.load(split_path)
        self.split_idx = loaded["splits"][split_num][
            "val"
        ]  # + loaded["splits"][split_num]["test"]
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
        idx = self.split_idx[i]
        eeg, label, image, subject = self.dataset.get_record(idx)
        return eeg, label, subject
