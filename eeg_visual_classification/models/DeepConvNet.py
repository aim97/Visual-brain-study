from torch import nn
from braindecode import models


class Model(nn.Module):
    """Deep ConvNet (Schirrmeister et al., 2017) via braindecode's Deep4Net.
    A four-stage deep convolutional pathway (temporal+spatial conv followed by
    three narrow-kernel conv-pool blocks) - the "deep convolutional pathway"
    baseline family Reviewer 2 flagged as missing."""

    def __init__(self, n_classes=40):
        super(Model, self).__init__()
        self.core = models.Deep4Net(
            n_outputs=n_classes,
            n_chans=128,
            n_times=440,
            sfreq=1000,
            input_window_seconds=0.44,
        )

    def forward(self, x):
        return self.core(x.squeeze(1))
