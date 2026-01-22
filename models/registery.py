# eeg_visual_classification/models/registry.py
from .EEGNET import Model as EEGNet
from .blstm import Model as blstm
from .BrainDecoder import Model as BrainDecoder
from .BrainDecoder3D import Model as BrainDecoder3D
from .lstm import Model as lstm
from .SleepingPower import Model as SleepingPower
from .EEGChannelNet import Model as EEGChannelNet
from .DCTVIT import Model as DCTVIT
from .CNN_LSTM import Model as CNN_LSTM
from .TemporalMap import Model as TemporalMap
from .EEGConformer import Model as EEGConformer
from .HumbleBrainDecoder3D import Model as HumbleBrainDecoder3D
from .SpectralBrainDecoder3D import Model as SpectralBrainDecoder3D
from .ATCNet import Model as ATCNet

MODEL_REGISTRY = {
    "EEGNET": EEGNet,
    "BLSTM": blstm,
    "BrainDecoder": BrainDecoder,
    "BrainDecoder3D": BrainDecoder3D,
    "lstm": lstm,
    "SleepingPower": SleepingPower,
    "EEGChannelNet": EEGChannelNet,
    "DCTVIT": DCTVIT,
    "CNN_LSTM": CNN_LSTM,
    "TemporalMap": TemporalMap,
    "EEGConformer": EEGConformer,
    "Hydra": HumbleBrainDecoder3D,
    "SpectralBrainDecoder3D": SpectralBrainDecoder3D,
    "ATCNet": ATCNet,
}
