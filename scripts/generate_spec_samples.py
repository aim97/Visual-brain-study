import torch
import matplotlib.pyplot as plt


from ..utils.lib import readSignal
from ..models.VisualTransforms import LogPowerSpectrum, LogWaveletCWT


data_path = "eeg_visual_classification/data/block/eeg_55_95_std.pth"
data = torch.load(data_path)
signal = readSignal(data, recordNo=10, channelNo=10)
signal = torch.from_numpy(signal)

# compute spectrogram with LogPowerSpectrum
log_power_spectrogram = (
    LogPowerSpectrum()(signal.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
)

# compute CWT with LogWaveletCWT
log_cwt = (
    LogWaveletCWT((812.5 / torch.linspace(5, 95, 100)).tolist())(
        signal.unsqueeze(0).unsqueeze(0)
    )
    .squeeze(0)
    .squeeze(0)
)

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(10, 4), dpi=150)

axs[0].imshow(
    log_power_spectrogram.numpy(),
    aspect="auto",
    origin="lower",
    cmap="jet",
)
axs[0].set_title("Log Power Spectrogram (5-95 Hz)")
axs[0].set_xlabel("Time Frames")
axs[0].set_ylabel("Frequency Bins")
axs[1].imshow(
    log_cwt.numpy(),
    aspect="auto",
    origin="lower",
    cmap="jet",
)
axs[1].set_title("Log Wavelet CWT (5-95 Hz)")
axs[1].set_xlabel("Time Frames")
axs[1].set_ylabel("Scales (Frequency Bins)")
plt.tight_layout()
plt.show()
