
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, nb_classes=40, chans=128, samples=440, dropout_rate=0.5,
                 kern_length=128, F1=8, D=2, F2=16):
        super(Model, self).__init__()

        # Block 1: Temporal + Depthwise
        self.conv_temporal = nn.Conv2d(1, F1, (1, kern_length), padding=(0, kern_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(F1, F1 * D, (chans, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout_rate)

        # Block 2: Separable Conv
        self.separable_conv = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        )
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout_rate)

        # Dynamically compute feature size
        self._to_linear = None
        self._get_conv_output(chans, samples)

        # Classification layer
        self.fc = nn.Linear(self._to_linear, nb_classes)

    def _get_conv_output(self, chans, samples):
        with torch.no_grad():
            x = torch.zeros(1, 1, chans, samples)
            x = self.conv_temporal(x)
            x = self.bn1(x)
            x = self.depthwise(x)
            x = self.bn2(x)
            x = F.elu(x)
            x = self.pool1(x)
            x = self.drop1(x)

            x = self.separable_conv(x)
            x = self.bn3(x)
            x = F.elu(x)
            x = self.pool2(x)
            x = self.drop2(x)

            self._to_linear = x.numel()

    def forward(self, x):
        x = self.conv_temporal(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.separable_conv(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = x.view(x.size(0), -1)
        return self.fc(x)

