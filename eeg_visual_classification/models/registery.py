import inspect

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
from .ShallowConvNet import Model as ShallowConvNet
from .DeepConvNet import Model as DeepConvNet
from .EEGInception import Model as EEGInception
from .EEGITNet import Model as EEGITNet

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
    "ShallowConvNet": ShallowConvNet,
    "DeepConvNet": DeepConvNet,
    "EEGInception": EEGInception,
    "EEGITNet": EEGITNet,
}

# A handful of models spell the "number of output classes" constructor
# argument differently, or don't expose it at all (fixed internally to 40).
# This lets every script pass `n_classes=` uniformly without needing to know
# each model's own convention.
_N_CLASSES_ALIASES = {
    "EEGNET": "nb_classes",
    "BLSTM": "num_classes",
    "EEGChannelNet": "num_classes",
}


def instantiate_model(name, **kwargs):
    """Build a registered model, adapting `n_classes` to whatever kwarg name
    the target model actually accepts, or dropping it (with a warning) if the
    model has a fixed, hardcoded class count.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model {name!r} not found in MODEL_REGISTRY. "
            f"Available: {sorted(MODEL_REGISTRY)}"
        )
    ModelClass = MODEL_REGISTRY[name]
    kwargs = dict(kwargs)

    if "n_classes" in kwargs:
        target_param = _N_CLASSES_ALIASES.get(name, "n_classes")
        accepted = inspect.signature(ModelClass.__init__).parameters
        if target_param != "n_classes":
            kwargs[target_param] = kwargs.pop("n_classes")
        elif "n_classes" not in accepted:
            n_classes = kwargs.pop("n_classes")
            print(
                f"[instantiate_model] {name!r} has a fixed class count and does "
                f"not accept n_classes; requested n_classes={n_classes} was ignored."
            )

    return ModelClass(**kwargs)
