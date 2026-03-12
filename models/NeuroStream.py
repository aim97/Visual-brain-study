import torch
import torch.nn as nn

from .meta.electrode_names import channels
from .VisualTransforms import EEGScalpMap


class R2Plus1D_Block(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        k_t=3,
        k_s=3,
        stride_t=1,
        stride_s=1,
        p_t=1,
        p_s=1,
        is_temporal_first=True,
        activation="SiLU",
    ):
        super().__init__()
        self.is_temporal_first = is_temporal_first

        # Build conv1 and conv2 so that conv1 always takes in_ch and conv2 takes out_ch
        if self.is_temporal_first:
            # temporal -> spatial
            self.conv1 = nn.Conv3d(
                in_ch,
                out_ch,
                kernel_size=(k_t, 1, 1),
                stride=(stride_t, 1, 1),
                padding=(p_t, 0, 0),
                bias=False,
            )
            self.conv2 = nn.Conv3d(
                out_ch,
                out_ch,
                kernel_size=(1, k_s, k_s),
                stride=(1, stride_s, stride_s),
                padding=(0, p_s, p_s),
                bias=False,
            )
        else:
            # spatial -> temporal
            self.conv1 = nn.Conv3d(
                in_ch,
                out_ch,
                kernel_size=(1, k_s, k_s),
                stride=(1, stride_s, stride_s),
                padding=(0, p_s, p_s),
                bias=False,
            )
            self.conv2 = nn.Conv3d(
                out_ch,
                out_ch,
                kernel_size=(k_t, 1, 1),
                stride=(stride_t, 1, 1),
                padding=(p_t, 0, 0),
                bias=False,
            )

        # A single BN after the two factored convs (keeps your original structure)
        self.bn = nn.BatchNorm3d(out_ch)
        self.act = nn.SiLU() if activation == "SiLU" else nn.ReLU()

        # Residual projection if channel or stride mismatch.
        # Overall stride is still (stride_t, stride_s, stride_s) regardless of op order.
        self.down = None
        if stride_t > 1 or stride_s > 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv3d(
                    in_ch,
                    out_ch,
                    kernel_size=1,
                    stride=(stride_t, stride_s, stride_s),
                    bias=False,
                ),
                nn.BatchNorm3d(out_ch),
            )

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        out = self.bn(out)
        identity = x if self.down is None else self.down(x)
        return self.act(out + identity)


class EEG3DNet(nn.Module):

    def __init__(
        self,
        n_classes=40,
        base=32,
        n_blocks=4,
        include_pooling=True,
        use_stem=True,
        activation="SiLU",
        is_temporal_first=True,
        video_channels=1,
        conv3d_kernel_size=(3, 3, 3),
        r21plus1d_spatial_kernel_size=3,
        r21plus1d_temporal_kernel_size=3,
    ):
        super().__init__()
        self.use_stem = use_stem
        self.include_pooling = include_pooling

        # ---- Stem ----
        self.stem = (
            nn.Sequential(
                nn.Conv3d(
                    video_channels,
                    base,
                    kernel_size=conv3d_kernel_size,
                    padding=(1, 1, 1),
                    bias=False,
                ),
                nn.BatchNorm3d(base),
                nn.SiLU() if activation == "SiLU" else nn.ReLU(),
            )
            if use_stem
            else nn.Identity()
        )

        # ---- R(2+1)D blocks ----
        blocks = []
        last_out_ch = (
            0  # if not use_stem else base  # initialize to stem output channels
        )
        for i in range(n_blocks):
            # Correct in_channels for first block:
            #   - if use_stem=True: stem outputs 'base' channels → first block input = base
            #   - if use_stem=False: raw input is (B,1,T,H,W) → first block input = 1
            in_channels = (
                base
                if use_stem and i == 0
                else (
                    last_out_ch if i > 0 else (video_channels if not use_stem else base)
                )
            )

            # Your original schedule for out_channels:
            out_channels = base * (i + (i % 2) + int(i == 0))
            last_out_ch = out_channels

            blocks.append(
                R2Plus1D_Block(
                    in_channels,
                    out_channels,
                    stride_t=i % 2 + 1,
                    stride_s=2,
                    k_s=r21plus1d_spatial_kernel_size,
                    k_t=r21plus1d_temporal_kernel_size,
                    activation=activation,
                    is_temporal_first=is_temporal_first,
                )
            )
        self.rds = nn.Sequential(*blocks)

        # ---- Head ----
        self.head = nn.Sequential()
        if include_pooling:
            self.head.append(nn.AdaptiveAvgPool3d((1, 1, 1)))
        self.head.append(nn.Flatten())

        # ---- Classifier input dim ----
        if include_pooling:
            # After GAP, the feature dim equals the last block's out_channels
            in_features = last_out_ch
        else:
            # If no pooling, output size depends on (T, H, W) after strides → infer with a dummy forward
            with torch.no_grad():
                # Choose a reasonable dummy input with enough T to survive temporal strides (e.g., T=32)
                dummy = torch.zeros(1, 1, 32, 32, 32)
                features = self.forward_features(dummy)
                in_features = features.shape[-1]

        self.classifier = nn.Linear(in_features, n_classes)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.rds(x)
        return self.head(x)

    def forward(self, x):  # x: (B,1,T,32,32)
        x = self.forward_features(x)
        return self.classifier(x)


class Model(nn.Module):

    def __init__(
        self,
        n_classes=40,
        features_only=False,
        n_blocks=4,
        include_pooling=True,
        use_stem=True,
        activation="SiLU",
        is_temporal_first=True,
    ):
        super().__init__()
        scalp_map_csv = "eeg_visual_classification/models/meta/map_v2.csv"
        self.map = EEGScalpMap(scalp_map_csv, electrodes=channels)
        self.model = EEG3DNet(
            n_classes,
            n_blocks=n_blocks,
            include_pooling=include_pooling,
            use_stem=use_stem,
            activation=activation,
            is_temporal_first=is_temporal_first,
        )
        self.features_only = features_only

    def forward(self, x):
        x = self.map(x.squeeze())  # (B, T, H, W)
        x = x.unsqueeze(1)  # (B, 1, T, H
        return self.model.forward_features(x) if self.features_only else self.model(x)

    @torch.no_grad()
    def forward_features(self, x):
        x = self.map(x.squeeze())  # (B, T, H, W)
        x = x.unsqueeze(1)  # (B, 1, T, H
        return self.model.forward_features(x)
