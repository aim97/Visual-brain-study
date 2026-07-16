import torch
import torch.nn as nn


class VisualOnlyClassifier(nn.Module):
    def __init__(self, in_features=768, out_classes=40):
        super(VisualOnlyClassifier, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_features, 512), nn.Linear(512, out_classes)
        )

    def forward(self, x):
        return self.seq.forward(x)


class SimpleClassifier(nn.Module):
    def __init__(self, layers=[768, 512], out_classes=40):
        super(SimpleClassifier, self).__init__()
        self.seq = nn.Sequential(
            *[
                nn.Linear(in_features, out_features)
                for in_features, out_features in zip(layers[:-1], layers[1:])
            ],
            nn.Linear(layers[-1], out_classes)
        )

    def forward(self, x):
        return self.seq.forward(x)
