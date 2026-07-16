import torch.nn as nn
from braindecode import models

class Model (nn.Module):
    def __init__(self):
      super(Model, self).__init__()
      self.core = models.SyncNet(
        n_outputs=40,
        n_chans=128,
        n_times=440,
        sfreq=1000,
        input_window_seconds=0.44,
      )
    def forward(self, x):
      return self.core(x.squeeze(1))