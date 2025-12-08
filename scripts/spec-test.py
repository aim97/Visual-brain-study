import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
from torcheeg import transforms

import torch
import torch.nn.functional as F
import pywt

from ..models import EEGViT


# Load EEG data
readRecord = lambda data, recordNo: data["dataset"][recordNo]["eeg"].numpy()
readSignal = lambda data, recordNo, channelNo: readRecord(data, recordNo)[channelNo]

dataset = "eeg_visual_classification/data/block/eeg_5_95_std.pth"
data = torch.load(dataset)

record = readRecord(data, 15)[:, 20:460]
signal = record[28]

fs = 1000
samples = signal.shape[0]
windowSize = 208
overlap = 180

def compute_spectrogram_db(signal: torch.Tensor, window_size=windowSize, overlap=overlap, fs=fs, freq_range=(5, 95)):
    window = torch.hann_window(window_size, periodic=True)
    stft_result = torch.stft(
        signal, 
        n_fft=window_size,
        hop_length=window_size - overlap,
        win_length=window_size,
        window=window, 
        return_complex=True, 
        center=False,
        normalized=False,
        onesided=True
    )
    magnitude = torch.abs(stft_result)
    # Convert to power and normalize
    power = (magnitude ** 2) / (window.norm() ** 2)
    spectrogram_db = 10 * torch.log10(power + 1e-12)
    print("Computed spectrogram shape:", spectrogram_db.shape)
    # Crop frequency range
    freqs = torch.linspace(0, fs / 2, window_size // 2 + 1)
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    return spectrogram_db[mask, :]

def modelspec(x):
    model = EEGViT.Model()
    return model.preview_image(torch.tensor(x))


# Create figure with 3 subplots
fig, axs = plt.subplots(2, 3, figsize=(10, 6), dpi=150)
plt.subplots_adjust(wspace=0.4)

# --- 1. Matplotlib Spectrogram ---
spec, frequencies, times, Sxx_matplotimg = axs[0][0].specgram(
    signal,
    NFFT=windowSize,
    Fs=fs,
    noverlap=overlap,
    window=np.hanning(windowSize),
    scale='dB',
    scale_by_freq=False,
    detrend='none'
)


# axs[0, 0].pcolormesh(time_edges, freq_edges, 10 * np.log10(spec + 1e-12), shading='gouraud', cmap='viridis')
axs[0, 0].set_title("Matplotlib Spectrogram")
axs[0, 0].set_xlabel("Time [s]")
axs[0, 0].set_ylabel("Frequency [Hz]")
axs[0, 0].set_ylim(0, 100)

print("Spectrogram shape (Matplotlib): ", spec.shape)
print("Image shape (Matplotlib specgram): ", Sxx_matplotimg.get_array().shape)

# --- 2. SciPy Spectrogram ---
frequencies, times, Sxx = sig.spectrogram(
    signal,
    fs=fs,
    window='hann',
    noverlap=overlap,
    nperseg=windowSize,
    nfft=windowSize,
    scaling='spectrum',
    mode='psd',
    detrend=False
)
pcm = axs[0][1].pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
axs[0, 1].set_title("SciPy Spectrogram")
axs[0, 1].set_xlabel("Time [s]")
axs[0, 1].set_ylabel("Frequency [Hz]")
axs[0, 1].set_ylim(0, 100)

print("Spectrogram shape (SciPy): ", Sxx.shape)
print("Image shape (SciPy pcolormesh): ", pcm.get_array().shape)

# --- 3. TorchEEG CWT Spectrogram ---
transform = transforms.CWTSpectrum(sampling_rate=fs, total_scale=100, wavelet='morl')
spectrogram = transform(eeg=record)['eeg']
img = axs[1, 0].imshow(10 * np.log10(np.abs(spectrogram[28])+1e-12), aspect='auto', origin='lower',
                    extent=[0, samples/fs, 0, 100], cmap='viridis')
axs[1, 0].set_title("TorchEEG CWT Spectrogram")
axs[1, 0].set_xlabel("Time [s]")
axs[1, 0].set_ylabel("Frequency [Hz]")
axs[1, 0].set_ylim(0, 100)

print("Spectrogram shape (TorchEEG CWT): ", spectrogram.shape)
print("Image shape (TorchEEG CWT): ", img.get_array().shape)

# 4 my implementation of Spectrogram
spectrogram = compute_spectrogram_db(torch.from_numpy(signal)).squeeze().float().numpy()
my_img = axs[1, 1].imshow(spectrogram, aspect='auto', origin='lower',
                    extent=[0, samples/fs, 0, 100], cmap='viridis')
axs[1, 1].set_title("My Spectrogram")
axs[1, 1].set_xlabel("Time [s]")
axs[1, 1].set_ylabel("Frequency [Hz]")
axs[1, 1].set_ylim(0, 100)

print("Spectrogram shape (My implementation): ", spectrogram.shape)
print("Image shape (My implementation): ", my_img.get_array().shape) 


# Define scales to cover 1–100 Hz
frequencies = np.arange(1, 101)
scales = pywt.scale2frequency('morl', frequencies) * fs

# Compute CWT
coeffs, freqs = pywt.cwt(signal, scales, 'morl', sampling_period=1/fs)
cwt_img_data =  10 * np.log10(np.abs(coeffs) + 1e-6)
cwt_img = axs[0, 2].imshow(cwt_img_data, aspect='auto', origin='lower',
                           extent=[0, len(signal)/fs, 1, 100], cmap='viridis')

# Plot spectrogram
axs[0, 2].set_title("Wavelet Spectrogram (1–100 Hz)")
axs[0, 2].set_xlabel("Time [s]")
axs[0, 2].set_ylabel("Frequency [Hz]")
axs[0, 2].set_ylim(0, 100)

print("Spectrogram shape (Wavelet CWT): ", cwt_img_data.shape)
print("Image shape (Wavelet CWT): ", cwt_img.get_array().shape)

# Model spectrogram
spectrogram = modelspec(record).squeeze().cpu().numpy()
my_other_img = axs[1, 2].imshow(spectrogram[28], aspect='auto', origin='lower', extent=[0, samples/fs, 0, 100], cmap='viridis')
axs[1, 2].set_title("My other Spectrogram")
axs[1, 2].set_xlabel("Time [s]")
axs[1, 2].set_ylabel("Frequency [Hz]")
axs[1, 2].set_ylim(0, 100)


# Add colorbars
fig.colorbar(Sxx_matplotimg, ax=axs[0, 0], label="Intensity [dB]")
fig.colorbar(pcm, ax=axs[0, 1], label="Intensity [dB]")
fig.colorbar(img, ax=axs[1 , 0], label="Intensity [dB]")
fig.colorbar(my_img, ax=axs[1 , 1], label="Intensity [dB]")
fig.colorbar(cwt_img, ax=axs[0 , 2], label="Intensity")
fig.colorbar(my_other_img, ax=axs[1 , 2], label="Intensity")


plt.tight_layout()
plt.show()