"""EEG datasets with
class for dataset dataloader
"""

import torch


# Dataset class
class EEGDataset:
    """EEG dataset

    Returns:
        EEG dataset: some data source
    """

    # Constructor
    def __init__(self, opt):
        # Load EEG signals
        loaded = torch.load(opt.eeg_dataset)  # signals_path)
        self.opt = opt
        if opt.subject != 0:
            self.data = [
                loaded["dataset"][i]
                for i in range(len(loaded["dataset"]))
                if loaded["dataset"][i]["subject"] == opt.subject
            ]
        else:
            self.data = loaded["dataset"]
        self.labels = loaded["labels"]
        self.images = loaded["images"]

        # print dataset sizes
        print(f"EEG dataset size: {len(self.data)}")
        print(f"Labels size: {len(self.labels)}")
        print(f"Images size: {len(self.images)}")

        # Compute size
        self.size = len(self.data)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Process EEG
        eeg = self.data[i]["eeg"].float().t()
        eeg = eeg[self.opt.time_low : self.opt.time_high, :]

        if self.opt.model_type == "model10":
            eeg = eeg.t()
            eeg = eeg.view(1, 128, self.opt.time_high - self.opt.time_low)
        # Get label
        label = self.data[i]["label"]
        # Return
        return eeg, label

    def get_record(self, i):
        """_summary_

        Args:
            i (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Process EEG
        eeg = self.data[i]["eeg"].float().t()
        eeg = eeg[self.opt.time_low : self.opt.time_high, :]

        if self.opt.model_type == "model10":
            eeg = eeg.t()
            eeg = eeg.view(1, 128, self.opt.time_high - self.opt.time_low)
        # Get label
        label = self.data[i]["label"]
        # Get image
        image = self.data[i]["image"]
        # Return
        return eeg, label, image
