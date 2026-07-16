from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import math
from scipy.fftpack import dct
from .meta.electrode_names import channels
from .VisualTransforms import EEGScalpMap

SCALP_MAP_CSV = str(Path(__file__).parent / "meta" / "map_v2.csv")

# Hyperparameters
num_classes = 10
image_size = 20
patch_size = 6
projection_dim = 64
dct_projection_dim = 64
num_heads = 4
transformer_units = [projection_dim * 2, projection_dim]
transformer_layers = 8
mlp_head_units = [2048, 1024]


# DCT-like transform using FFT
def dct_2d_torch(x):
    X = torch.fft.fft2(x)
    return X.real


# Patch extraction
class Patches(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=(patch_size, patch_size), stride=patch_size)

    def forward(self, x):
        patches = self.unfold(x)  # (B, C*patch_size*patch_size, num_patches)
        patches = patches.transpose(1, 2)  # (B, num_patches, patch_dim)
        return patches


# Patch encoder (dynamic)
class PatchEncoder(nn.Module):
    def __init__(self, patch_dim, num_patches, projection_dim, dct_projection_dim):
        super().__init__()
        self.projection = nn.Linear(patch_dim, projection_dim)
        self.dct_projection = nn.Linear(patch_dim, dct_projection_dim)
        self.position_embedding = nn.Parameter(torch.randn(num_patches, projection_dim))

    def forward(self, patch, dct_patch):
        encoded = (
            self.projection(patch)
            + self.dct_projection(dct_patch)
            + self.position_embedding
        )
        return encoded


# MLP block
def make_mlp(input_dim, hidden_units, dropout_rate):
    layers = []
    for units in hidden_units:
        layers.append(nn.Linear(input_dim, units))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout_rate))
        input_dim = units
    return nn.Sequential(*layers)


# Transformer block
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_units):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=0.1, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = make_mlp(embed_dim, mlp_units, 0.1)

    def forward(self, x):
        x1 = self.norm1(x)
        attn_output, _ = self.attn(x1, x1, x1)
        x2 = x + attn_output
        x3 = self.norm2(x2)
        x3 = self.mlp(x3)
        return x2 + x3


# Full STTM model
class STTMClassifier(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.patches = Patches(patch_size)
        num_patches = (image_size // patch_size) ** 2
        patch_dim = patch_size * patch_size * in_channels
        self.encoder = PatchEncoder(
            patch_dim, num_patches, projection_dim, dct_projection_dim
        )
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(projection_dim, num_heads, transformer_units)
                for _ in range(transformer_layers)
            ]
        )
        self.norm = nn.LayerNorm(projection_dim)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.mlp_head = make_mlp(
            num_patches * projection_dim, mlp_head_units + [num_classes], 0.5
        )

    def forward(self, x):
        patches = self.patches(x)
        dct_map = dct_2d_torch(x)
        dct_patches = self.patches(dct_map)
        encoded = self.encoder(patches, dct_patches)
        for block in self.transformer_blocks:
            encoded = block(encoded)
        representation = self.norm(encoded)
        representation = self.flatten(representation)
        representation = self.dropout(representation)
        return self.mlp_head(representation)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.map = EEGScalpMap(SCALP_MAP_CSV, electrodes=channels)
        self.model = STTMClassifier(440)

    def forward(self, x):
        x = x.squeeze(1)  # remove channel dim if exists
        x = self.map(x)  # map EEG to scalp images
        # resize to (20,20)
        x = F.interpolate(
            x, size=(image_size, image_size), mode="bilinear", align_corners=False
        )
        return self.model(x)
