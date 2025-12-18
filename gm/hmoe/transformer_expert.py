import torch
import torch.nn.functional as F
from torch import nn

from gm.hmoe.multi_head_attention import MultiHeadAttention


class TransformerExpert(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=3072, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Pre-LN architecture
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.self_attn = MultiHeadAttention(
            d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        # FFN implemented via nn.Parameter
        self.ffn_W1 = nn.Parameter(torch.empty(dim_feedforward, d_model))
        self.ffn_b1 = nn.Parameter(torch.zeros(dim_feedforward))
        self.ffn_W2 = nn.Parameter(torch.empty(d_model, dim_feedforward))
        self.ffn_b2 = nn.Parameter(torch.zeros(d_model))

        self.dropout = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.ffn_W1, mean=0.0, std=0.02)
        nn.init.normal_(self.ffn_W2, mean=0.0, std=0.02)
        nn.init.zeros_(self.ffn_b1)
        nn.init.zeros_(self.ffn_b2)

    def forward(self, x, attn_mask=None):
        # Self-attention with Pre-LN
        x_norm = self.norm1(x)
        attn_output = self.self_attn(
            x_norm,
            attn_mask=attn_mask,
        )
        x = x + self.dropout(attn_output)

        # Feed-forward network with Pre-LN
        x_norm = self.norm2(x)
        ff_output = F.linear(x_norm, self.ffn_W1, self.ffn_b1)
        ff_output = F.gelu(ff_output)
        ff_output = self.ffn_dropout(ff_output)
        ff_output = F.linear(ff_output, self.ffn_W2, self.ffn_b2)
        x = x + self.dropout(ff_output)

        return x
