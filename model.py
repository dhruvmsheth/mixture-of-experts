import torch
import torch.nn as nn
from moe import MoE

class LanguageModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, k=4):
        super(LanguageModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)  # Use a linear layer for embedding
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.moe = MoE(input_size=hidden_dim, output_size=hidden_dim, num_experts=num_experts, k=k)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Embedding layer
        embed = self.embedding(x)
        # First LSTM layer
        out, _ = self.lstm1(embed)
        out = out + embed  # Residual connection

        # Reshape for MoE layer
        batch_size, seq_len, hidden_dim = out.size()
        out_reshaped = out.reshape(-1, hidden_dim)  # Use reshape instead of view

        # MoE layer
        moe_out_reshaped, aux_loss = self.moe(out_reshaped)
        moe_out = moe_out_reshaped.reshape(batch_size, seq_len, hidden_dim)  # Reshape back
        moe_out = torch.sigmoid(moe_out)
        moe_out = self.dropout(moe_out)
        moe_out = moe_out + out  # Residual connection

        # Second LSTM layer
        out, _ = self.lstm2(moe_out)
        out = out + moe_out  # Residual connection

        # Output layer
        out = self.dropout(out)
        logits = self.fc(out)
        # Return logits and auxiliary loss from MoE
        return logits, aux_loss