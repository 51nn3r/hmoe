from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from gm.hmoe.experts_storage import ExpertsStorage
from gm.utils.masking import extend_mask_for_learnable_vectors


class HMoeComponent(nn.Module):
    '''Base interface for MoE components'''

    gate: nn.Module | None
    experts_storage: ExpertsStorage | None
    chain_size: int | None
    prev_hmoe: HMoeComponent | None
    prev_hmoe_chian_size: int | None
    trainable_vectors: nn.Parameter | None

    def __init__(
            self,
            gate: Optional['nn.Module'] = None,
            experts_storage: Optional['ExpertsStorage'] = None,
            chain_size: Optional['int'] = None,
            d_model: int = 32,
    ):
        super().__init__()

        self.gate = gate
        self.experts_storage = experts_storage
        self.chain_size = chain_size
        # The HMoE for which this component acts as a gate
        self.prev_hmoe = None
        self.prev_hmoe_chian_size = None
        self.trainable_vectors = None

        self.d_model = d_model

        self._built = False

    def _build(self, x):
        '''Initialize internal layers based on the input shape'''
        self._built = True

    def _extend_x_and_mask_with_trainable_vectors(self, x, attn_mask):
        if self.trainable_vectors is None:
            return x, attn_mask

        trainable_batch = self.trainable_vectors.unsqueeze(0).expand(x.size(0), -1, -1)
        x = torch.cat([trainable_batch, x], dim=1)
        if attn_mask is not None:
            attn_mask = extend_mask_for_learnable_vectors(attn_mask, self.trainable_vectors.size(0))

        return x, attn_mask

    def set_prev_hmoe(self, prev: HMoeComponent):
        # self.prev_hmoe = prev
        self.prev_hmoe_chian_size = prev.chain_size

        self.trainable_vectors = nn.Parameter(torch.randn(prev.chain_size, self.d_model))  # @TODO: fix self.d_model
        nn.init.xavier_uniform_(self.trainable_vectors)

    def forward(self, x):
        if not self._built:
            self._build(x)
