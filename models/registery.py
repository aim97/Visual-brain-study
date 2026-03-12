# eeg_visual_classification/models/registry.py
from .EEGNET import Model as EEGNet
from .blstm import Model as blstm
from .BrainDecoder import Model as BrainDecoder
from .NeuroStream import Model as NeuroStream
from .lstm import Model as lstm
from .SleepingPower import Model as SleepingPower
from .EEGChannelNet import Model as EEGChannelNet
from .DCTVIT import Model as DCTVIT
from .CNN_LSTM import Model as CNN_LSTM
from .TemporalMap import Model as TemporalMap
from .EEGConformer import Model as EEGConformer
from .FusedBrainDecoder3D import Model as FusedBrainDecoder
from .NeuroStream4D import Model as NeuroStream4D
from .ATCNet import Model as ATCNet

MODEL_REGISTRY = {
    "EEGNET": EEGNet,
    "BLSTM": blstm,
    "BrainDecoder": BrainDecoder,
    "NeuroStream": NeuroStream,
    "lstm": lstm,
    "SleepingPower": SleepingPower,
    "EEGChannelNet": EEGChannelNet,
    "DCTVIT": DCTVIT,
    "CNN_LSTM": CNN_LSTM,
    "TemporalMap": TemporalMap,
    "EEGConformer": EEGConformer,
    "FusedBrainDecoder": FusedBrainDecoder,
    "NeuroStream4D": NeuroStream4D,
    "ATCNet": ATCNet,
}
