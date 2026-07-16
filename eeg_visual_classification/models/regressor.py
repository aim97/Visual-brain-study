import torch
from torch import nn


class SimpleRegressor(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, in_features=768, out_classes=40):
        super(SimpleRegressor, self).__init__()
        self.seq = nn.Sequential(nn.Linear(in_features, out_classes))

    def forward(self, x):
        """_summary_

        Args:
            x (torch.Tensor): input tensor
        """
        return self.seq.forward(x)


class RegressorV1(nn.Module):
    def __init__(self, input_size, output_size):
        super(RegressorV1, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)  # Batch normalization after the first layer
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)  # Batch normalization after the second layer
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        # Input normalization
        x = (x - torch.mean(x, dim=0)) / torch.std(x, dim=0)

        x = self.fc1(x)
        # x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        # x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        return x
