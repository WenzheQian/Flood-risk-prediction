import torch
import torch.nn as nn

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, seq_length):
        super().__init__()
        self.input_proj = nn.Linear(input_size, 4)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=4, 
            nhead=4,
            dim_feedforward=256
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.fc = nn.Linear(seq_length * 4, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = x.permute(1, 0, 2)  # [seq_len, batch, features]
        output = self.transformer(x)
        output = output.permute(1, 0, 2).reshape(x.shape[1], -1)
        return self.fc(output)