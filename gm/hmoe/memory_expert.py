import torch
from torch import nn

from gm.hmoe.transformer_expert import TransformerExpert
from gm.utils.masking import extend_mask_for_memory_vectors

"""
MemoryExpert extends a standard Transformer expert with a small set of learnable
memory vectors.

These vectors are appended as additional time steps to the input sequence and
participate in self-attention, allowing the expert to store and retrieve persistent
information. After the Transformer block, the memory positions are removed, so the
output sequence length matches the original input.

When memory_vectors_count = 0, this class is strictly equivalent to a standard
Transformer expert, enabling clean ablation studies.
"""


class MemoryExpert(nn.Module):
    memory_vectors_count: int
    d_model: int
    transformer_expert: TransformerExpert
    memory_vectors: nn.Parameter | None

    def __init__(
            self,
            memory_vectors_count: int,
            d_model: int,
            num_heads: int,
            dim_feedforward: int = 3072,
            dropout: float = 0.1,
    ):
        super().__init__()

        self.memory_vectors_count = memory_vectors_count
        self.d_model = d_model

        self.transformer_expert = TransformerExpert(
            d_model=d_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        if memory_vectors_count > 0:
            self.memory_vectors = nn.Parameter(
                torch.empty(memory_vectors_count, d_model)
            )
            self.reset_parameters()
        else:
            self.register_parameter("memory_vectors", None)

    def reset_parameters(self):
        """
        Initialization follows Transformer conventions:
        small normal noise to avoid dominating early attention.
        """
        nn.init.normal_(self.memory_vectors, mean=0.0, std=0.02)
        # nn.init.zeros_(self.memory_vectors)

    def forward(
            self,
            x: torch.Tensor,
            attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x: [B, T, D]
        attn_mask: must already include memory tokens if present
        """
        if self.memory_vectors_count == 0:
            return self.transformer_expert(x, attn_mask=attn_mask)

        B, T, _ = x.shape

        # [M, D] -> [B, M, D]
        mem = self.memory_vectors.unsqueeze(0).expand(B, -1, -1)

        # concat memory as additional time steps
        x_aug = torch.cat([x, mem], dim=1)  # [B, T+M, D]

        # print(attn_mask)
        # print(extend_mask_for_memory_vectors(attn_mask, self.memory_vectors_count))
        out = self.transformer_expert(x_aug, attn_mask=extend_mask_for_memory_vectors(attn_mask, self.memory_vectors_count))

        # slice back original time steps
        return out[:, :T, :]  # [B, T, D]
