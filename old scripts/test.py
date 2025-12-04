
import torch
import torchaudio
import braindecode
import mne

print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Braindecode:", braindecode.__version__)
print("MNE:", mne.__version__)
print("Torchaudio:", torchaudio.__version__)
