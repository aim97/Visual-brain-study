import numpy as np
import torch
import torchvision.transforms as TV
import torchaudio.transforms as TA
import torch.nn as nn
import scipy.signal as sig
from functools import reduce 
from math import isnan

import torch
import torch.nn.functional as F

def compute_spectrogram(signal: torch.Tensor, window_size: int = 128, overlap: int = 96) -> torch.Tensor:
    """
    Computes the spectrogram of a 1D signal using a Hanning window.

    Args:
        signal (torch.Tensor): 1D input signal tensor.
        window_size (int): Size of the window (default: 128).
        overlap (int): Number of overlapping samples (default: 96).

    Returns:
        torch.Tensor: Spectrogram (magnitude) of shape [num_frames, window_size // 2 + 1].
    """
    # Ensure signal is 1D
    if signal.dim() != 1:
        raise ValueError("Input signal must be a 1D tensor.")

    # Create Hanning window
    window = torch.hann_window(window_size, periodic=True)

    # Compute STFT
    stft_result = torch.stft(
        signal,
        n_fft=window_size,
        hop_length=window_size - overlap,
        window=window,
        return_complex=True
    )

    # Compute magnitude
    spectrogram = torch.abs(stft_result)

    return spectrogram

class LambdaLayer(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.l = fn
    def forward(self, x):
        return self.l(x)

def tab(marker):
    def display(x):
        print(f"{marker} {x.shape}")
        return x
    return display

Tab = lambda x: LambdaLayer(tab(x))

class MLP_layer(nn.Module):
    def __init__(self, in_dim, out_dim, activation=None):
        super(MLP_layer, self).__init__()
        self.model = nn.Linear(in_dim, out_dim)
        if activation == None:
            self.activation = lambda x: x
        else:
            self.activation = activation

    def forward(self, x):
        return self.activation(self.model(x))

def get_power_spectrum(signal):
    fs = 100
    f, t, Zxx = sig.stft(
        signal[0], 
        fs=fs, nperseg=200, noverlap=100, 
        window=np.hamming(200),
        nfft=256
    )
    Zxx = torch.from_numpy(Zxx).float()
    stft_log_power = torch.log(torch.abs(Zxx) ** 2)
    stft_log_power_normalized = stft_log_power - torch.mean(stft_log_power)
    # C:\Users\moham\lab\AttnSleep\utils\util.py:48:  VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
    return stft_log_power_normalized

def generate_power_spectrum_torch(signal):
    # print(f"signal shape: {signal.shape}")
    Zxxs = torch.stft(
        signal.squeeze(),
        n_fft=128,
        win_length=128,
        hop_length=32,
        window=torch.hamming_window(128).cuda(),
        return_complex=True,
    )
    stft_log_power = torch.log(torch.abs(Zxxs) ** 2 + 1e-13)
    # isNanX = reduce(lambda x, y: x or y, torch.isnan(signal).cpu().numpy().flatten(), False)
    # isNanImageSide = reduce(lambda x, y: x or y, torch.isnan(Zxxs).cpu().numpy().flatten(), False)
    # if (not isNanX and isNanImageSide):
    #     print("from image")
    #     print(Zxxs)
    # 3 - Normalize the log power spectrum
    stft_log_power_normalized = stft_log_power - torch.mean(stft_log_power)
    l = len(stft_log_power_normalized.shape)
    # print("shape length: ", l)
    if l == 3:
        ret = LambdaLayer(lambda x: x[:, None, :, :])(stft_log_power_normalized)
    elif l == 2:
        ret = LambdaLayer(lambda x: x[None, None, :, :])(stft_log_power_normalized)
    else:
        raise ValueError("Expected length of 2 or 3, found %d" % l)
    # print(f"output image shape: {ret.shape}")
    return ret

def get_power_spectrum_batch(signal):
    def window_function(window_length):
        return torch.hamming_window(window_length).cuda()
    fs = 100
    # print(f"==> {signal.shape}")
    Zxxs = []
    signal.to(device)
    for i in range(signal.shape[0]):
        f, t, Zxx = sig.stft(
            signal[i][0], 
            fs=fs, 
            nperseg=200, 
            noverlap=100, 
            window=window_function, 
            nfft=256
        )
        Zxxs.append(Zxx)
    print(f"==> {len(Zxxs)}")

    Zxxs = torch.from_numpy(np.array(Zxxs)).float()
    # print("zxxs shape", Zxxs.shape)
    # 2 - Convert to log power spectrum
    stft_log_power = torch.log(torch.abs(Zxxs) ** 2)
    # 3 - Normalize the log power spectrum
    stft_log_power_normalized = stft_log_power - torch.mean(stft_log_power)
    return stft_log_power_normalized