from torch import nn
from braindecode import models


class Model(nn.Module):
    """Shallow ConvNet (Schirrmeister et al., 2017) via braindecode's
    ShallowFBCSPNet. A wide-filter-bank, single-stage "multi-scale" temporal
    convolution followed by a spatial filter and log-variance pooling -
    a shallow-and-wide counterpart to the deep, narrow-kernel architectures
    (EEGChannelNet, DeepConvNet) already used as baselines here."""

    def __init__(self, n_classes=40):
        super(Model, self).__init__()
        self.core = models.ShallowFBCSPNet(
            n_outputs=n_classes,
            n_chans=128,
            n_times=440,
            sfreq=1000,
            input_window_seconds=0.44,
        )

    def forward(self, x):
        return self.core(x.squeeze(1))
