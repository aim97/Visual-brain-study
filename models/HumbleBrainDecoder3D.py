import torch
import torch.nn as nn

from .SleepingPower import Model as VisualProcessor
from .BrainDecoder3D import Model as ScalpReader


class Model(nn.Module):
    def __init__(self, n_classes=40):
        super().__init__()
        self.visualProcessor = VisualProcessor(features_only=True)
        self.ScalpReader = ScalpReader(features_only=True)

        self.compress = nn.Linear(256, 128)

        self.classifier = nn.Sequential(nn.Linear(128, 64), nn.Linear(64, n_classes))

    def forward(self, x):
        visual_features = self.compress(self.visualProcessor(x))
        scalp_features = self.ScalpReader(x)
        # print(visual_features.shape)
        # print(scalp_features.shape)
        features = visual_features + scalp_features
        return self.classifier(features)
