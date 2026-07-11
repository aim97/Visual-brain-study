import argparse
from ..models import MODEL_REGISTRY
from torchinfo import summary
from eeg_visual_classification.utils.lib import (
    extract_model_options,
)

parser = argparse.ArgumentParser(description="Summarize model")

# Model type/options
parser.add_argument(
    "-mt",
    "--model_type",
    default="lstm",
    help="specify which generator should be used: lstm|EEGChannelNet",
)
parser.add_argument(
    "-mp",
    "--model_params",
    default="",
    nargs="*",
    help="list of key=value pairs of model options",
)

opt = parser.parse_args()

ModelClass = MODEL_REGISTRY.get(opt.model_type)
if ModelClass is None:
    raise ValueError(f"Model {opt.model_type} not found in MODEL_REGISTRY.")
model_options = extract_model_options(opt.model_params)
model = ModelClass(**model_options)

# input_size format: (batch_size, channels/electrodes, sequence_length)
summary(model, input_size=(16, 128, 440))
