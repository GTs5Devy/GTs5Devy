import torch
import torch.nn as nn

class HandlingLSTM(nn.Module):
    def __init__(self, input_size=12, hidden_size=64, num_layers=2, output_size=3, dropout=0.3):
        """
        input_size: 12 — matches the 12 physics features in dataset_lstm.py
        hidden_size: 64 — number of LSTM memory cells
        num_layers: 2 — stacked LSTM layers
        output_size: 3 — Neutral, Understeer, Oversteer
        dropout: 0.3 — helps prevent overfitting on limited lap data
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM — dropout only applies between layers (not on last layer)
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Dropout before final classification layer
        self.dropout = nn.Dropout(dropout)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [batch_size, sequence_length, input_size]
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))  # LSTM forward pass
        out = out[:, -1, :]              # take output from last timestep only
        out = self.dropout(out)          # apply dropout
        out = self.fc(out)               # classify
        return out