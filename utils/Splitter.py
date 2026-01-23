import torch
from torch.utils.data import Dataset


# Splitter class
class Splitter(Dataset):
    """Dataset splitter

    Args:
        Dataset (_type_): _description_
    """

    def __init__(
        self,
        dataset,
        split_path,
        split_num=0,
        split_name="train",
        is_semantic=False,
        combined_test_val=True,
        return_full_record=False,
    ):
        # Set EEG dataset
        self.dataset = dataset
        self.return_full_record = return_full_record
        # Load split
        loaded = torch.load(split_path)
        self.split_idx = (
            loaded[split_name]
            if is_semantic
            else loaded["splits"][split_num][split_name]
        )
        if combined_test_val and split_name == "test":
            # append val indices for test set
            self.split_idx += loaded["splits"][split_num]["val"]
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
        eeg, label, image, subject, _ = self.dataset.get_record(self.split_idx[i])
        # Return
        if self.return_full_record:
            return eeg, label, image, subject
        else:
            return eeg, label
