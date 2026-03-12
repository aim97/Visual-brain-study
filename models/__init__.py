from .AttnSleep import Model as AttnSleep
from .EEGNET import Model as EEGNET
from .SleepingPower import Model as SleepingPower
from .visualModels import ViT, SimpleEEGCNN, ResizeTo64, ResNet18Wrapper
from .BrainDecoder import Model as BrainDecoder
from .NeuroStream import Model as NeuroStream
from .SleepingPower import LogPowerSpectrum, LogWaveletCWT
from .blstm import Model as blstm
from .EEGChannelNet import Model as EEGChannelNet
from .lstm import Model as lstm
from .EEGConformer import Model as EEGConformer
from .NeuroStream4D import Model as NeuroStream4D

from .registery import MODEL_REGISTRY
