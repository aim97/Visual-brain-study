
import torch
import torch.nn as nn

# EEG_BiLSTM
class Model(nn.Module):
    def __init__(self, input_size=128, hidden_size=128, num_layers=2, num_classes=40, dropout=0.3):
        super(Model, self).__init__()
        
        # BiLSTM
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional
        
    def forward(self, x):
        # x shape: (batch, channels, samples)
        # Reshape to (batch, seq_len, features)
        x = x.squeeze(1)  # Remove channel dimension if it's 1
        x = x.permute(0, 2, 1)  # (batch, seq_len=samples, features=channels)
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        
        # Use last time step
        out = lstm_out[:, -1, :]  # (batch, hidden*2)
        
        # Fully connected
        out = self.fc(out)  # (batch, num_classes)
        
        return out
