from pathlib import Path

import torch
import torch.nn as nn
from .meta.electrode_names import channels
from .VisualTransforms import EEGScalpMap

SCALP_MAP_CSV = str(Path(__file__).parent / "meta" / "map_v2.csv")


class FrameCNN(nn.Module):
    def __init__(self, feat_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.SiLU(),  # 32->16
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.SiLU(),  # 16->8
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, feat_dim),
        )

    def forward(self, x):  # (B*T,1,32,32)
        return self.net(x)


class TemporalBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, d=1, p_drop=0.1):
        super().__init__()
        pad = (k - 1) // 2 * d
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, k, padding=pad, dilation=d),
            nn.BatchNorm1d(c_out),
            nn.SiLU(),
            nn.Dropout(p_drop),
            nn.Conv1d(c_out, c_out, k, padding=pad, dilation=d),
            nn.BatchNorm1d(c_out),
        )
        self.act = nn.SiLU()
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        return self.act(self.net(x) + self.down(x))


class TCN(nn.Module):
    def __init__(self, in_ch, channels=(256, 256, 256, 256), k=3, p_drop=0.1):
        super().__init__()
        layers, cprev = [], in_ch
        for i, c in enumerate(channels):
            layers.append(TemporalBlock(cprev, c, k=k, d=2**i, p_drop=p_drop))
            cprev = c
        self.net = nn.Sequential(*layers)
        self.out_ch = cprev

    def forward(self, x):
        return self.net(x)


class CNN_TCN_Classifier(nn.Module):
    def __init__(self, n_classes=40, feat_dim=128, tcn_channels=(256,) * 4):
        super().__init__()
        self.frame = FrameCNN(feat_dim=feat_dim)
        self.tcn = TCN(in_ch=feat_dim, channels=tcn_channels)
        self.cls = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(self.tcn.out_ch, n_classes)
        )

    def forward(self, x):  # x: (B,1,T,32,32)
        B, C, T, H, W = x.shape
        f = self.frame(x.permute(0, 2, 1, 3, 4).reshape(B * T, 1, H, W))  # (B*T, feat)
        f = f.view(B, T, -1).transpose(1, 2)  # (B, feat, T)
        z = self.tcn(f)  # (B, c, T)
        return self.cls(z)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.map = EEGScalpMap(SCALP_MAP_CSV, electrodes=channels)
        self.classifier = CNN_TCN_Classifier(n_classes=40)

    def forward(self, x):
        x = self.map(x.squeeze())  # (B, T, H, W)
        x = x.unsqueeze(1)  # (B, 1, T, H, W)
        x = self.classifier(x)
        return x
