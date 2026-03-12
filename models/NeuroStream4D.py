import torch
import torch.nn as nn
import numpy as np

from .NeuroStream import EEG3DNet
from .meta.electrode_names import channels
from .VisualTransforms import SpectralEEGScalpMap, LogWaveletCWT, LogPowerSpectrum


class Model(nn.Module):

    def __init__(
        self,
        spec_type="cwt",
        n_classes=40,
        bin_size=10,
        base=32,
        features_only=False,
        n_blocks=4,
        include_pooling=True,
        use_stem=True,
        activation="SiLU",
        is_temporal_first=True,
        conv3d_kernel_size=(3, 3, 3),
        r21plus1d_spatial_kernel_size=3,
        r21plus1d_temporal_kernel_size=3,
    ):
        super().__init__()
        scalp_map_csv = "eeg_visual_classification/models/meta/map_v2.csv"
        self.power_transform = (
            LogWaveletCWT((812.5 / np.linspace(5, 95, 100)).tolist(), bin_size=bin_size)
            if spec_type == "cwt"
            else LogPowerSpectrum(fmax_hz=100)
        )
        self.map = SpectralEEGScalpMap(scalp_map_csv, electrodes=channels)
        self.model = EEG3DNet(
            n_classes,
            n_blocks=n_blocks,
            include_pooling=include_pooling,
            use_stem=use_stem,
            activation=activation,
            is_temporal_first=is_temporal_first,
            video_channels=(
                (100 + bin_size - 1) // bin_size if spec_type == "cwt" else 10
            ),
            conv3d_kernel_size=conv3d_kernel_size,
            r21plus1d_spatial_kernel_size=r21plus1d_spatial_kernel_size,
            r21plus1d_temporal_kernel_size=r21plus1d_temporal_kernel_size,
            base=base,
        )
        self.features_only = features_only
        self.bin_size = bin_size

    def forward(self, x):  # -> Any:  # x: (B, 1, F, T)
        x = self.power_transform(x.squeeze())  # B, C, F', T
        # B, C, F, T = x.shape
        # B, C, F'', T
        # print("After power transform: ", x.shape)
        # x = x.view(B, C, F // self.bin_size, self.bin_size, T).mean(dim=3)
        # print("After aggregation: ", x.shape)
        x = x.permute(0, 2, 3, 1)  # B, F'', T, C
        # print("After reshaping: ", x.shape)
        x = self.map(x.squeeze())  # (B, F'', T, H, W)
        # print("After ScalpMapping: ", x.shape)
        return self.model.forward_features(x) if self.features_only else self.model(x)

    @torch.no_grad()
    def forward_features(self, x):
        x = self.power_transform(x.squeeze())  # B, C, F', T
        # B, C, F, T = x.shape
        # B, C, F'', T
        # print("After power transform: ", x.shape)
        # x = x.view(B, C, F // self.bin_size, self.bin_size, T).mean(dim=3)
        # print("After aggregation: ", x.shape)
        x = x.permute(0, 2, 3, 1)  # B, F'', T, C
        # print("After reshaping: ", x.shape)
        x = self.map(x.squeeze())  # (B, F'', T, H, W)
        # print("After ScalpMapping: ", x.shape)
        return self.model.forward_features(x)
