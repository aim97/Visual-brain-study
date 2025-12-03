# -*- coding: utf-8 -*-
import math
from typing import Optional, Literal
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Preprocess: EEG -> (B, 128, H, W) spectrogram tensor ----------
class Spectrogram128(nn.Module):
    """
    Converts EEG (B, 128, T) to a 2D tensor (B, 128, H, W):
      - STFT per channel
      - Crop 5..95 Hz
      - Log magnitude
      - Resize to (out_size, out_size)
      - InstanceNorm per channel
    """

    def __init__(
        self,
        sample_rate: float = 1000.0,
        fmin: float = 5.0,
        fmax: float = 95.0,
        n_fft: int = 512,
        win_length: int = 128,
        hop_length: int = 32,
        out_size: int = 224,
        log_eps: float = 1e-12,
        instance_norm: bool = True,
    ):
        super().__init__()
        self.fs = float(sample_rate)
        self.fmin = float(fmin)
        self.fmax = float(fmax)
        self.n_fft = int(n_fft)
        self.win_length = int(win_length)
        self.hop_length = int(hop_length)
        self.out_size = int(out_size)
        self.log_eps = float(log_eps)

        # Normalization across spatial dims per channel (per sample)
        self.inst_norm = (
            nn.InstanceNorm2d(128, affine=False, eps=1e-5)
            if instance_norm
            else nn.Identity()
        )

    def _stft_mag(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T) -> |STFT|: (B, C, F, Tspec)
        """
        B, C, T = x.shape
        dev = x.device
        window = torch.hann_window(self.win_length, periodic=True, device=dev)
        x_flat = x.reshape(B * C, T)

        S = torch.stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=False,
            normalized=False,
            onesided=True,
            return_complex=True
        )
        mag = torch.abs(S)  # (B*C, F, Tspec)
        power = (mag ** 2) / (window.norm() ** 2)
        power_db = 10 * torch.log10(power + self.log_eps)
        # print("STFT magnitude shape:", power_db.shape)
        Freq, Tspec = power_db.shape[-2], power_db.shape[-1]
        power_db = power_db.view(B, C, Freq, Tspec)
        # print("Reshaped STFT magnitude shape:", power_db.shape)
        return power_db

    def _freq_crop(self, spec: torch.Tensor) -> torch.Tensor:
        """
        spec: (B, C, F, Tspec) -> crop 5..95 Hz
        """
        B, C, F, Tspec = spec.shape
        freqs = torch.linspace(0.0, self.fs / 2.0, self.win_length // 2 + 1, device=spec.device)
        mask = (freqs >= self.fmin) & (freqs <= self.fmax)
        if not mask.any():
            # Fallback: keep nearest bin to fmin
            idx = (freqs - max(self.fmin, 0)).abs().argmin()
            mask = torch.zeros_like(freqs, dtype=torch.bool)
            mask[idx] = True
        return spec[:, :, mask, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 128, 440) or (128, 440)
        returns: (B, 128, out_size, out_size)
        """
        if x.dim() == 2:  # (C, T)
            x = x.unsqueeze(0)
        assert (
            x.dim() == 3 and x.shape[1] == 128
        ), f"Expected (B,128,T), got {tuple(x.shape)}"

        mag = self._stft_mag(x)  # (B,128,F,Tspec)
        mag = self._freq_crop(mag)  # (B,128,Fcrop,Tspec)
        # print("Cropped STFT magnitude shape:", mag.shape)

        # Log compression (stable for small magnitudes)
        # img = torch.log(mag + self.log_eps)

        # Resize to square for ViT patching
        img = F.interpolate(
            mag,
            size=(self.out_size, self.out_size),
            mode="bilinear",
            align_corners=False,
        )

        # Instance normalization per channel
        img = self.inst_norm(img)  # (B,128,H,W)
        return img


# ---------- Variant A: timm ViT with in_chans=128 (recommended for 8GB) ----------
class EEGViT128_Timm(nn.Module):
    """
    ViT-Small/16 from timm with in_chans=128. Pretrained=False (since stem differs).
    """

    def __init__(self, num_classes: int, img_size: int = 224):
        super().__init__()
        try:
            import timm
        except Exception as e:
            raise ImportError("Please `pip install timm` to use this variant.") from e

        # Smaller model fits 8GB easily; you can also try vit_tiny_patch16_224 for even lighter
        self.vit = timm.create_model(
            "vit_small_patch16_224",
            pretrained=False,  # stem is 128-ch; pretrained weights are for 3-ch
            in_chans=128,
            num_classes=num_classes,
            img_size=img_size,
        )

    def forward(self, x128: torch.Tensor) -> torch.Tensor:
        return self.vit(x128)


# ---------- Variant B: torchvision ViT-B/16 with 128-ch patch embedding ----------
class EEGViT128_Torchvision(nn.Module):
    """
    Modifies torchvision vit_b_16 to accept 128 input channels by replacing the patch embedding conv.
    Pretrained=False by default (stem changes). You *can* load pretrained and inflate weights (optional).
    """

    def __init__(
        self,
        num_classes: int,
        img_size: int = 224,
        pretrained: bool = False,
        inflate_from_rgb: bool = False,
    ):
        super().__init__()
        try:
            from torchvision.models import vit_b_16, ViT_B_16_Weights
        except Exception as e:
            raise ImportError("torchvision vision_transformer not available.") from e
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.vit = vit_b_16(weights=weights)
        self.vit.heads.head = nn.Linear(self.vit.heads.head.in_features, num_classes)

        # Replace 3-ch patch embedding conv with 128-ch
        old = self.vit.conv_proj  # Conv2d(3 -> embed_dim, k=16, s=16)
        new = nn.Conv2d(
            in_channels=128,
            out_channels=old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=(old.bias is not None),
        )

        if pretrained and inflate_from_rgb:
            # Inflate 3-channel weights to 128-channel stem (simple heuristic)
            with torch.no_grad():
                W = old.weight  # [embed_dim, 3, 16, 16]
                # Mean across RGB, then repeat to 128; scale to preserve variance
                Wm = W.mean(dim=1, keepdim=True)  # [embed_dim,1,16,16]
                W128 = Wm.repeat(1, 128, 1, 1) * (
                    3.0 / 128.0
                )  # scale so sum energy ~ similar
                new.weight.copy_(W128)
                if old.bias is not None:
                    new.bias.copy_(old.bias)
        else:
            # Default init
            pass

        self.vit.conv_proj = new

    def forward(self, x128: torch.Tensor) -> torch.Tensor:
        return self.vit(x128)


# ---------- End-to-end wrapper ----------
class Model(nn.Module):
    """
    Full pipeline:
      EEG (B,128,T=440) -> Spectrogram128 -> (B,128,224,224) -> ViT -> logits
    """

    def __init__(
        self,
        num_classes: int = 40,
        backend: Literal["timm_small", "torchvision_base"] = "torchvision_base",
        sample_rate: float = 1000.0,
        # STFT params suited for fs=1000 and T≈440
        n_fft: int = 256,
        win_length: int = 256,
        hop_length: int = 32,
        fmin: float = 5.0,
        fmax: float = 95.0,
        image_size: int = 224,
    ):
        super().__init__()
        self.pre = Spectrogram128(
            sample_rate=sample_rate,
            fmin=fmin,
            fmax=fmax,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            out_size=image_size,
            instance_norm=True,
        )

        if backend == "timm_small":
            self.backbone = EEGViT128_Timm(num_classes=num_classes, img_size=image_size)
        elif backend == "torchvision_base":
            self.backbone = EEGViT128_Torchvision(
                num_classes=num_classes, img_size=image_size, pretrained=False
            )
        else:
            raise ValueError("backend must be 'timm_small' or 'torchvision_base'")

    @torch.no_grad()
    def preview_image(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns preprocessed (B,128,224,224) tensor for visualization.
        """
        return self.pre(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img128 = self.pre(x.squeeze())  # (B,128,224,224)
        logits = self.backbone(img128)
        return logits

readSignal = lambda data, recordNo: data["dataset"][recordNo]["eeg"].numpy()

# -------- Example usage --------
if __name__ == "__main__":
    B, C, T = 8, 128, 440
    data = torch.load("data/block/eeg_55_95_std.pth")
    x = readSignal(data, 15)[:, 20: 460]  # (128, 440)

    model = Model()

    with torch.cuda.amp.autocast(False):
        # preview preprocessed image
        img128 = model.preview_image(torch.tensor(x))  # (1,128,224,224)
        print("Preprocessed image:", img128.shape)
        plt.imshow(img128[0, 28].cpu(), cmap="viridis")
        plt.title("Preprocessed Spectrogram Channel 0")
        plt.colorbar()
        plt.show()
