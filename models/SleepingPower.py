import torch.nn as nn
from ..models.visualModels import ViT, SimpleEEGCNN
from ..models.VisualTransforms import LogPowerSpectrum, LogWaveletCWT
import numpy as np
from typing import Optional, Literal


class Model(nn.Module):
    def __init__(
        self,
        spec_type: Literal["stft", "cwt"] = "cwt",
        core_model: Literal["ViT", "ResNet18", "SimpleEEGCNN"] = "SimpleEEGCNN",
        resize: Optional[int] = None,
    ):
        super().__init__()
        self.powerImage = (
            LogPowerSpectrum()
            if spec_type == "stft"
            else LogWaveletCWT((812.5 / np.linspace(5, 95, 100)).tolist())
        )
        # self.resize = ResizeTo64()
        if core_model == "ViT":
            if resize is not None:
                img_dim = resize
                batch_dim = 16
            elif spec_type == "cwt":
                img_dim = 100
                batch_dim = 10
            else:
                img_dim = 224
                batch_dim = 16

            self.visual_model = ViT(
                img_dim=img_dim,
                in_channels=128,
                patch_dim=batch_dim,
                num_classes=40,
                dim=32,
                blocks=1,
                heads=32,
                dim_linear_block=32,
                classification=True,
                classification_structure=True,
            )
        elif core_model == "ResNet18":
            from models.visualModels import ResNet18Wrapper

            self.visual_model = ResNet18Wrapper(num_classes=40)
        elif core_model == "SimpleEEGCNN":
            self.visual_model = SimpleEEGCNN(num_classes=40)
        else:
            raise ValueError("core_model must be 'ViT', 'ResNet18', or 'SimpleEEGCNN'")

        self.resize = (
            None
            if resize is None
            else nn.Upsample(size=(resize, resize), mode="bilinear")
        )

    def forward(self, x):
        x = self.powerImage(x.squeeze())
        if self.resize is not None:
            x = self.resize(x)
        x = self.visual_model(x)

        return x
