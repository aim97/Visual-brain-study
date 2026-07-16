"""Print a torchinfo layer-by-layer summary for a registered model.
Run directly: python eeg_visual_classification/scripts/summarize_model.py -mt NeuroStream4D
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from torchinfo import summary

from eeg_visual_classification.models import instantiate_model
from eeg_visual_classification.utils.lib import extract_model_options

parser = argparse.ArgumentParser(description="Summarize model")

# Model type/options
parser.add_argument(
    "-mt",
    "--model_type",
    default="lstm",
    help="specify which generator should be used: lstm|EEGChannelNet|NeuroStream4D|...",
)
parser.add_argument(
    "-mp",
    "--model_params",
    default="",
    nargs="*",
    help="list of key=value pairs of model options",
)

opt = parser.parse_args()

model_options = extract_model_options(opt.model_params)
model = instantiate_model(opt.model_type, **model_options)

# input_size format: (batch_size, channel, electrodes, sequence_length)
summary(model, input_size=(16, 1, 128, 440))
