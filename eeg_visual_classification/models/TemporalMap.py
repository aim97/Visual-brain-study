from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from .meta.electrode_names import channels
from .VisualTransforms import EEGScalpMap

SCALP_MAP_CSV = str(Path(__file__).parent / "meta" / "map_v2.csv")


def conv_2plus1d(
    in_ch, out_ch, k_t=3, k_s=3, stride_t=1, stride_s=1, padding_t=1, padding_s=1
):
    # Spatial 2D conv
    spatial = nn.Conv3d(
        in_ch,
        out_ch,
        kernel_size=(1, k_s, k_s),
        stride=(1, stride_s, stride_s),
        padding=(0, padding_s, padding_s),
        bias=False,
    )
    # Temporal 1D conv (across time)
    temporal = nn.Conv3d(
        out_ch,
        out_ch,
        kernel_size=(k_t, 1, 1),
        stride=(stride_t, 1, 1),
        padding=(padding_t, 0, 0),
        bias=False,
    )
    return spatial, temporal


class R2Plus1DBlock(nn.Module):
    def __init__(
        self, in_ch, out_ch, stride_t=1, stride_s=1, downsample=None, dropout=0.0
    ):
        super().__init__()
        self.spatial1, self.temporal1 = conv_2plus1d(
            in_ch, out_ch, stride_t=stride_t, stride_s=stride_s
        )
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.spatial2, self.temporal2 = conv_2plus1d(out_ch, out_ch)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.downsample = downsample
        self.dropout = nn.Dropout3d(dropout)

    def forward(self, x):
        identity = x
        out = self.spatial1(x)
        out = self.temporal1(out)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.dropout(out)

        out = self.spatial2(out)
        out = self.temporal2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = F.relu(out + identity, inplace=True)
        return out


class TemporalPyramid(nn.Module):
    """Multi-scale temporal pooling without losing spatial resolution."""

    def __init__(self, scales=(1, 2, 4)):
        super().__init__()
        self.scales = scales

    def forward(self, x):
        # x: (B, C, T, H, W)
        outs = []
        for s in self.scales:
            if s == 1:
                outs.append(x)
            else:
                outs.append(F.avg_pool3d(x, kernel_size=(s, 1, 1), stride=(s, 1, 1)))
        # Align along T by adaptive pooling to min T among branches
        minT = min(o.shape[2] for o in outs)
        outs = [
            F.adaptive_avg_pool3d(o, output_size=(minT, x.shape[3], x.shape[4]))
            for o in outs
        ]
        return torch.cat(outs, dim=1)  # concat on channel


class R2Plus1D_Lite(nn.Module):
    """
    Input: (B, T, C=1, H=15, W=11) -> permuted to (B, C, T, H, W)
    """

    def __init__(self, num_classes=10, base_ch=32, dropout=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(
                1,
                base_ch,
                kernel_size=(3, 3, 3),
                stride=(2, 1, 1),
                padding=(1, 1, 1),
                bias=False,
            ),  # T: 440 -> 220
            nn.BatchNorm3d(base_ch),
            nn.ReLU(inplace=True),
        )
        # Stage 1: 220 -> 110 (temporal stride 2)
        self.layer1 = R2Plus1DBlock(
            base_ch,
            base_ch,
            stride_t=2,
            stride_s=1,
            downsample=nn.Sequential(
                nn.Conv3d(
                    base_ch, base_ch, kernel_size=1, stride=(2, 1, 1), bias=False
                ),
                nn.BatchNorm3d(base_ch),
            ),
            dropout=dropout,
        )

        # Stage 2: channel up, keep T
        self.layer2 = R2Plus1DBlock(
            base_ch,
            base_ch * 2,
            stride_t=1,
            stride_s=1,
            downsample=nn.Sequential(
                nn.Conv3d(base_ch, base_ch * 2, kernel_size=1, bias=False),
                nn.BatchNorm3d(base_ch * 2),
            ),
            dropout=dropout,
        )

        # Stage 3: modest channels
        self.layer3 = R2Plus1DBlock(
            base_ch * 2, base_ch * 2, stride_t=1, stride_s=1, dropout=dropout
        )

        # Temporal Pyramid to aggregate multi-scale
        self.tpp = TemporalPyramid(scales=(1, 2, 4))  # concat channels x3

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),  # GAP over T,H,W
            nn.Flatten(),
            nn.Linear(base_ch * 2 * 3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.tpp(x)
        x = self.head(x)
        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.map = EEGScalpMap(SCALP_MAP_CSV, electrodes=channels)
        self.model = R2Plus1D_Lite(num_classes=40, base_ch=32, dropout=0.3)

    def forward(self, x):
        x = x.squeeze(1)  # emove channel dim if exists
        x = self.map(x)  # map EEG to scalp images
        x = x.unsqueeze(2)
        return self.model(x)
