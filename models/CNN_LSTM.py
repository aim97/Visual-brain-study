import torch
import torch.nn as nn

from .meta.electrode_names import channels
from .VisualTransforms import EEGScalpMap


# EEG_CNN_LSTM
class EEG_CNN_LSTM(nn.Module):
    def __init__(self, num_classes=10, hidden_size=128):
        super(EEG_CNN_LSTM, self).__init__()

        # CNN for spatial feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 15x15 -> 7x7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7 -> 3x3
        )

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(input_size=384, hidden_size=hidden_size, batch_first=True)

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x shape: (batch, time, 1, 15, 15)
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        spatial_features = self.cnn(x)  # (b*t, 64, 3, 3)
        spatial_features = spatial_features.view(b, t, -1)  # (batch, time, features)

        lstm_out, _ = self.lstm(spatial_features)
        last_hidden = lstm_out[:, -1, :]  # take last time step
        return self.fc(last_hidden)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        scalp_map_csv = "eeg_visual_classification/models/meta/map_v2.csv"
        self.map = EEGScalpMap(scalp_map_csv, electrodes=channels)

        self.video_model = EEG_CNN_LSTM(num_classes=40)

    def forward(self, x):
        x = x.squeeze(1)  # remove channel dim if exists
        x = self.map(x)  # map EEG to scalp images
        x = x.unsqueeze(2)
        return self.video_model(x)
