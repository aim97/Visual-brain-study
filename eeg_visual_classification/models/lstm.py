# Original model presented in: C. Spampinato, S. Palazzo, I. Kavasidis, D. Giordano, N. Souly, M. Shah, Deep Learning Human Mind for Automated Visual Classification, CVPR 2017
import sys
import os
import random
import math
import time
import torch

torch.utils.backcompat.broadcast_warning.enabled = True
from torchvision import transforms, datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
import numpy as np


class Model(nn.Module):

    def __init__(
        self,
        n_classes=40,
        input_size=128,
        lstm_size=128,
        lstm_layers=1,
        output_size=128,
    ):
        # Call parent
        super().__init__()
        # Define parameters
        self.input_size = input_size
        self.lstm_size = lstm_size
        self.lstm_layers = lstm_layers
        self.output_size = output_size

        # Define internal modules
        self.lstm = nn.LSTM(
            input_size, lstm_size, num_layers=lstm_layers, batch_first=True
        )
        self.output = nn.Linear(lstm_size, output_size)
        self.classifier = nn.Linear(output_size, n_classes)

    def forward(self, x):
        # Prepare LSTM initiale state
        batch_size = x.size(0)
        lstm_init = (
            torch.zeros(self.lstm_layers, batch_size, self.lstm_size, device=x.device),
            torch.zeros(self.lstm_layers, batch_size, self.lstm_size, device=x.device),
        )

        # Forward LSTM and get final state
        x = self.lstm(x, lstm_init)[0][:, -1, :]

        # Forward output
        x = F.relu(self.output(x))
        x = self.classifier(x)
        return x
