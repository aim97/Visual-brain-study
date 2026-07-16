from torch import nn
from braindecode import models


class Model(nn.Module):
    """EEG-Inception (Santamaria-Vazquez et al., 2020) via braindecode's
    EEGInception. Stacks parallel convolutional branches with different
    kernel widths per block (an Inception-style "multi-branch structure")
    to capture temporal patterns at multiple scales at once - directly
    answers Reviewer 2's complaint about missing multi-branch EEG CNN
    baselines."""

    def __init__(self, n_classes=40):
        super(Model, self).__init__()
        # braindecode renamed EEGInception -> EEGInceptionERP (the visually
        # evoked / ERP-tuned variant, as opposed to EEGInceptionMI for motor
        # imagery) in more recent releases.
        self.core = models.EEGInceptionERP(
            n_outputs=n_classes,
            n_chans=128,
            n_times=440,
            sfreq=1000,
            input_window_seconds=0.44,
        )

    def forward(self, x):
        return self.core(x.squeeze(1))
