import numpy as np
import torch
import matplotlib.pyplot as plt

format = lambda x: x.replace("_", " ")
readSignal = lambda data, recordNo, channelNo: data["dataset"][recordNo]["eeg"][channelNo].numpy()

def prepare_frequencies(sampling_rate, duration):
  time = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
  # Compute FFT
  freq = np.fft.rfftfreq(len(time), 1 / (sampling_rate))
  return (freq, time)

def generate_frequency_distribution(signal, mask):
  # Compute FFT
  fft_amplitude = np.abs(np.fft.rfft(signal))
  fft_amplitude = fft_amplitude[mask]
  return fft_amplitude
  
  
basePath = "data/block"
  
datasets = [
  "eeg_signals_raw_with_mean_std",
  "eeg_55_95_std",
  "eeg_14_70_std",
  "eeg_5_95_std",
]

displayName = [
  "Raw signal",
  "55-95 Hz",
  "14-70 Hz",
  "5-95 Hz",
]

# freqs, time = prepare_frequencies(1000, 0.5)
# mask = freqs <= 100
# freqs = freqs[mask]

# Create a figure and a grid of subplots with 2 rows and 4 columns
fig, axs = plt.subplots(2, 4, figsize=(12, 6))

# Generate sample data and plot in each subplot
for j in range(4):
  dataset = datasets[j]
  datasetDisplayName = displayName[j]
  # Load data
  data = torch.load(f"{basePath}/{dataset}.pth")
  signal = readSignal(data, 15,28)
  time = np.linspace(0, 500, signal.shape[0], endpoint=False)
  
  # time domain plot
  ax = axs[0, j]
  ax.plot(time, signal)
  ax.set_xlim(0, 500)
  ax.set_title(datasetDisplayName)
  ax.set_xlabel("Time [s]")
  ax.set_ylabel("Amplitude")
  ax.grid()
  # frequency domain plot
  ax = axs[1, j]
  ax.psd(signal, NFFT=512, Fs=1000)
  ax.set_xlim(0, 130)
  ax.set_xlabel("Frequency [Hz]")
  ax.set_ylabel("Power [dB]")
  ax.grid()

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the figure
plt.show()