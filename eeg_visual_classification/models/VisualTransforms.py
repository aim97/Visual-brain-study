import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from .meta.electrode_names import channels


class LogWaveletCWT(nn.Module):
    def __init__(
        self, scales, wavelet="morl", eps=1e-8, mean_normalize=False, bin_size=None
    ):
        super().__init__()
        self.scales = scales
        self.eps = eps
        self.mean_normalize = mean_normalize
        self.bin_size = bin_size

        # Precompute wavelet kernels
        kernels = []
        length = 128  # kernel length (choose based on signal duration)
        t = torch.linspace(-length / 2, length / 2, steps=length)
        for s in scales:
            kernel = torch.exp(-(t**2) / (2 * s**2)) * torch.cos(5 * t / s)  # Morlet
            kernel = kernel / kernel.norm()  # normalize
            kernels.append(kernel)
        kernels = torch.stack(kernels).unsqueeze(1)  # (num_scales, 1, length)
        self.register_buffer("kernels", kernels)

    def forward(self, x):
        B, C, T = x.shape
        x = x.view(B * C, 1, T)
        coeffs = F.conv1d(x, self.kernels, padding=self.kernels.size(-1) // 2)
        coeffs = coeffs[..., :T]
        coeffs = coeffs.view(B, C, len(self.scales), coeffs.size(-1))
        log_power = torch.log(coeffs.abs().pow(2) + self.eps)
        if self.mean_normalize:
            log_power = log_power - log_power.mean(dim=(-2, -1), keepdim=True)

        if self.bin_size is not None:
            S = log_power.size(2)
            bs = int(self.bin_size)
            if bs <= 0:
                raise ValueError(f"bin_size must be > 0, got {bs}")

            chunks = torch.split(log_power, bs, dim=2)  # list of (B,C,bi,T)
            log_power = torch.stack(
                [c.mean(dim=2) for c in chunks], dim=2
            )  # (B,C,n_bins,T)

        return log_power


class LogPowerSpectrum(nn.Module):
    """
    A PyTorch layer that computes the log power spectrum via STFT for batched signals.

    Input:
        x: Tensor of shape (B, T) or (B, C, T)
    Output:
        log_power: Tensor of shape (..., F, K), where
            F = nfft // 2 + 1        (frequency bins, one-sided)
            K = number of frames (time steps)

    Parameters mirror SciPy stft-style args you used.
    """

    def __init__(
        self,
        fs: int = 1000,
        nperseg: int = 96,  # 128
        noverlap: int = int(96 * 0.9),  # 112
        nfft: int = 96,  # 128
        window: str = "hamming",  # or a precomputed torch.Tensor of length nperseg
        center: bool = False,  # SciPy pads by default; set to False to be close to your call
        pad_mode: str = "reflect",
        eps: float = 1e-8,
        mean_normalize: bool = True,
        per_sample: bool = False,  # False => global mean (matches your function)
        fmax_hz: (
            float | None
        ) = None,  # NEW: None => keep full band; else slice to [0, fmax_hz]
    ):
        super().__init__()

        hop_length = nperseg - noverlap
        if hop_length <= 0:
            raise ValueError(
                f"hop_length must be > 0, got nperseg={nperseg}, noverlap={noverlap}"
            )

        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft
        self.hop_length = hop_length
        self.center = center
        self.pad_mode = pad_mode
        self.eps = eps
        self.mean_normalize = mean_normalize
        self.per_sample = per_sample
        self.fmax_hz = fmax_hz

        # Build/register window as a buffer so it moves with .to(device)
        if isinstance(window, torch.Tensor):
            win = window
            if win.numel() != nperseg:
                raise ValueError(
                    f"Provided window has length {win.numel()}, expected {nperseg}."
                )
        else:
            w = window.lower()
            if w == "hamming":
                # periodic=True matches FFT usage (SciPy uses fftbins=True in STFT)
                win = torch.hamming_window(nperseg, periodic=True)
            elif w == "hann":
                win = torch.hann_window(nperseg, periodic=True)
            else:
                raise ValueError(
                    f"Unsupported window '{window}'. Use 'hamming', 'hann', or a Tensor."
                )
        self.register_buffer("window", win)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T) or (B, C, T)
        returns: (..., F, K) log power spectrum (optionally mean-normalized)
        """
        if x.ndim not in (2, 3):
            raise ValueError(
                f"Expected input of shape (B,T) or (B,C,T); got {tuple(x.shape)}"
            )

        # Ensure window dtype/device match input
        window = self.window.to(dtype=x.dtype, device=x.device)
        if x.ndim == 3:
            # (B, C, T) -> (B*C, T)
            B, C, T = x.shape
            x = x.reshape(B * C, T)

        # torch.stft supports batched inputs: (..., time) -> (..., freq, frames)
        Z = torch.stft(
            x,
            n_fft=self.nfft,
            hop_length=self.hop_length,
            win_length=self.nperseg,
            window=window,
            center=self.center,
            pad_mode=self.pad_mode,
            normalized=False,
            return_complex=True,  # complex tensor: (..., F, K)
        )

        # Power and log-power
        power = Z.abs().pow(2)  # (..., F, K)
        log_power = torch.log(power + self.eps)

        # --- NEW: frequency-band slicing [0, fmax_hz] ---
        if self.fmax_hz is not None:
            # k_max = floor(fmax_hz * nfft / fs), but clamp to last bin (nfft//2)
            max_bin = self.nfft // 2
            k_max = int(
                torch.clamp(
                    torch.tensor(
                        self.fmax_hz * self.nfft / self.fs, device=log_power.device
                    ),
                    min=0,
                    max=max_bin,
                )
                .floor()
                .item()
            )
            log_power = log_power[..., : (k_max + 1), :]  # keep bins 0..k_max

        # Mean normalization (default: global to match your function)
        if self.mean_normalize:
            if self.per_sample:
                # per-sample mean across freq & time
                mean = log_power.mean(dim=(-2, -1), keepdim=True)
            else:
                # global mean over the whole batch/tensor (your original behavior)
                mean = log_power.mean()
            log_power = log_power - mean

        # fix shape to (..., F, K)
        log_power = log_power.view(B, C, log_power.size(-2), log_power.size(-1))

        return log_power

    @property
    def freq_bins(self) -> int:
        return self.nfft // 2 + 1

    @property
    def freqs(self) -> torch.Tensor:
        """Frequency axis in Hz (sliced if fmax_hz is set)."""
        full = torch.fft.rfftfreq(self.nfft, d=1.0 / self.fs).to(self.window.device)
        if self.fmax_hz is None:
            return full
        k_max = min(int(self.fmax_hz * self.nfft // self.fs), self.nfft // 2)
        return full[: k_max + 1]

    def frame_times(self, num_frames: int) -> torch.Tensor:
        """Helper to get time axis (sec) given output frames."""
        return torch.arange(num_frames, device=self.window.device) * (
            self.hop_length / self.fs
        )


class EEGScalpMap(nn.Module):
    """
    PyTorch layer that maps EEG channel data onto a 2D scalp map grid.
    """

    def __init__(
        self, electrode_positions_csv: str, electrodes=channels, dtype=torch.float32
    ):
        super().__init__()
        if electrodes is None or len(electrodes) == 0:
            raise ValueError(
                "`electrodes` (list of channel names in the same order as x[:, :, :]) is required."
            )
        self.electrodes = list(electrodes)
        self.dtype = dtype

        # Load CSV and convert each cell to a channel index (or -1)
        df = pd.read_csv(electrode_positions_csv, header=None).fillna("")
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

        def to_idx(cell):
            if isinstance(cell, str) and cell in self.electrodes:
                return self.electrodes.index(cell)
            return -1

        idx_grid = df.map(to_idx).values  # (H, W) int array
        idx_grid = torch.as_tensor(idx_grid, dtype=torch.long)
        self.register_buffer("idx_grid", idx_grid, persistent=False)
        self.H, self.W = idx_grid.shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        returns: (B, T, H, W)
        """
        if x.ndim != 3:
            raise ValueError(f"x must be (B, C, T), got {tuple(x.shape)}")
        B, C, T = x.shape
        H, W = self.H, self.W

        # 1) Put time next to batch: (B, T, C)
        x_bt = x.transpose(1, 2).contiguous()

        # 2) Pad dummy zero-channel on the right → (B, T, C+1)
        zero_pad = torch.zeros(B, T, 1, dtype=x.dtype, device=x.device)
        x_bt_pad = torch.cat([x_bt, zero_pad], dim=2)  # channel C is the dummy

        # 3) Prepare safe indices: map -1 -> C (dummy channel)
        idx = self.idx_grid.to(x.device)  # (H, W)
        idx_safe = torch.where(idx < 0, torch.full_like(idx, C), idx)  # (H, W)

        # 4) Match dims for gather:
        #    - Lift data to 5D so we can gather along channel dim=2
        #      x5d: (B, T, C+1, H, W) by unsqueezing and expanding
        x5d = x_bt_pad.unsqueeze(-1).unsqueeze(-1)  # (B, T, C+1, 1, 1)
        x5d = x5d.expand(B, T, C + 1, H, W)  # (B, T, C+1, H, W)

        #    - Make indices 5D with a singleton channel dim: (B, T, 1, H, W)
        idx5d = idx_safe.view(1, 1, 1, H, W).expand(B, T, 1, H, W)

        # 5) Gather along the channel dimension (dim=2) → (B, T, 1, H, W)
        scalp = torch.gather(x5d, dim=2, index=idx5d)

        # 6) Squeeze the singleton channel dim → (B, T, H, W)
        scalp = scalp.squeeze(2)
        return scalp


class SpectralEEGScalpMap(nn.Module):
    r"""
    Map EEG channel data to a 2D scalp map grid.

    UPDATED (strict):
      * Accepts:
          - x: (B, T, C)          → returns (B, T, H, W)
          - x: (B, P, T, C)       → returns (B, P, T, H, W)
        where C == len(electrodes)
      * No internal handling of legacy layouts.

    Args:
        electrode_positions_csv: CSV whose cells contain channel names (or empty).
        electrodes: list of channel names in the same order as x[..., :, C]
        dtype: dtype for internal buffers

    Notes:
        Any CSV entry not found in `electrodes` is treated as -1 (filled with zeros).
    """

    def __init__(self, electrode_positions_csv: str, electrodes, dtype=torch.float32):
        super().__init__()
        if electrodes is None or len(electrodes) == 0:
            raise ValueError(
                "`electrodes` (list of channel names in the same order as the last dim of x) is required."
            )

        self.electrodes = list(electrodes)
        self.num_channels = len(self.electrodes)
        self.dtype = dtype

        # Load CSV grid and convert cell→channel index (or -1)
        df = pd.read_csv(electrode_positions_csv, header=None).fillna("")
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

        def to_idx(cell):
            if isinstance(cell, str) and cell in self.electrodes:
                return self.electrodes.index(cell)
            return -1  # empty/missing cell

        idx_grid = df.map(to_idx).values  # (H, W) int array
        idx_grid = torch.as_tensor(idx_grid, dtype=torch.long)
        self.register_buffer("idx_grid", idx_grid, persistent=False)
        self.H, self.W = idx_grid.shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape
                - (B, T, C)      or
                - (B, P, T, C)
            C must equal len(self.electrodes)

        Returns:
            scalp:
                - (B, T, H, W)       if input was (B, T, C)
                - (B, P, T, H, W)    if input was (B, P, T, C)
        """
        if x.ndim == 3:
            B, T, C = x.shape
            if C != self.num_channels:
                # Common mistake: legacy (B, C, T)
                if x.shape[1] == self.num_channels:
                    raise ValueError(
                        f"Expected x to be (B, T, C={self.num_channels}) but got (B, C, T). "
                        f"Transpose upstream: x = x.transpose(1, 2)."
                    )
                raise ValueError(
                    f"Last dim (C) must be {self.num_channels}, got {C}. "
                    f"Ensure x is (B, T, C) with channels matching `electrodes`."
                )
            # Normalize to (B, P, T, C) with P=1
            x = x.unsqueeze(1)  # (B, 1, T, C)
            squeeze_P = True

        elif x.ndim == 4:
            B, P, T, C = x.shape
            if C != self.num_channels:
                # Common mistake: legacy (B, P, C, T)
                if x.shape[2] == self.num_channels:
                    raise ValueError(
                        f"Expected x to be (B, P, T, C={self.num_channels}) but got (B, P, C, T). "
                        f"Transpose upstream: x = x.transpose(2, 3)."
                    )
                raise ValueError(
                    f"Last dim (C) must be {self.num_channels}, got {C}. "
                    f"Ensure x is (B, P, T, C) with channels matching `electrodes`."
                )
            squeeze_P = False
        else:
            raise ValueError(
                f"x must be 3D or 4D: (B,T,C) or (B,P,T,C), got {tuple(x.shape)}"
            )

        # From here x is (B, P, T, C) with correct C
        B, P, T, C = x.shape
        H, W = self.H, self.W

        # Pad a dummy zero-channel along the channel axis → (B, P, T, C+1)
        zero_pad = x.new_zeros((B, P, T, 1))
        x_bptc = torch.cat([x, zero_pad], dim=3)  # (B, P, T, C+1)
        # Prepare indices: -1 → C (dummy channel)
        idx = self.idx_grid.to(x.device)  # (H, W)
        idx_safe = torch.where(idx < 0, torch.full_like(idx, C), idx)  # (H, W)

        # Expand and gather over channel dimension (dim=3)
        # x6d:   (B, P, T, C+1, H, W)
        x6d = x_bptc.unsqueeze(-1).unsqueeze(-1).expand(B, P, T, C + 1, H, W)
        # idx6d: (B, P, T, 1,   H, W)
        idx6d = idx_safe.view(1, 1, 1, 1, H, W).expand(B, P, T, 1, H, W)
        # Gather → (B, P, T, 1, H, W) → squeeze channel → (B, P, T, H, W)
        scalp = torch.gather(x6d, dim=3, index=idx6d).squeeze(3)

        if squeeze_P:
            scalp = scalp.squeeze(1)  # (B, T, H, W)

        return scalp
