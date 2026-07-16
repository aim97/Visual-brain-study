from torch import nn
from braindecode import models


class Model(nn.Module):
    """EEG-ITNet (Salami et al., 2022) via braindecode's EEGITNet. Combines
    parallel Inception-style multi-branch temporal convolutions with
    depthwise-separable and dilated causal blocks, giving a lightweight
    multi-scale/multi-branch baseline distinct in design from EEGInception -
    part of the response to Reviewer 2's request for more multi-branch and
    multi-scale CNN baselines."""

    def __init__(self, n_classes=40):
        super(Model, self).__init__()
        self.core = models.EEGITNet(
            n_outputs=n_classes,
            n_chans=128,
            n_times=440,
            sfreq=1000,
            input_window_seconds=0.44,
        )

    def forward(self, x):
        return self.core(x.squeeze(1))
