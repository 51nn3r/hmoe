import torch
from torch import nn

from gm.hmoe.hmoe_component import HMoeComponent
from gm.hmoe.multi_head_attention import MultiHeadAttention


class DeepDarkGate(HMoeComponent):
    def __init__(
            self,
            d_model=None,
            key_dim=None,
            num_heads=8,
            dropout_rate=0.0,
            use_bias=True,
    ):
        super().__init__(d_model=d_model)

        self.mha = MultiHeadAttention(
            d_model=d_model,
            key_dim=key_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            use_bias=use_bias,
        )

        self.linear = nn.Linear(d_model, d_model)

    def _build(self, x):
        super()._build(x)

    def forward(self, x, attn_mask=None):
        super().forward(x)

        x, attn_mask = self._extend_x_and_mask_with_trainable_vectors(x, attn_mask)
        attn = self.mha(x, attn_mask=attn_mask)
        y = self.linear(attn)

        out = y[:, :self.trainable_vectors.size(0), :]

        return {
            'out': out,
            'gates_out': torch.sum(out[:, None, :, 1:], axis=-2),
            'exp_usage': None,
        }
