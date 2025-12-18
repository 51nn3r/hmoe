from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class HMoeUsage:
    weights: torch.Tensor
    top_ids: torch.Tensor
    prev: HMoeUsage | None

    def __init__(self, weights: torch.Tensor, top_ids, prev=None):
        self.weights = weights
        self.top_ids = top_ids
        self.prev = prev

    def kldiv_loss(self):
        loss = nn.KLDivLoss(reduction='batchmean')

        return loss(F.log_softmax(self.weights, dim=-1), self.correct_weights)

    @property
    def correct_weights(self):
        correct = torch.zeros_like(self.weights)

        batch_size, seq_len, k = self.top_ids.shape
        batch_idx = torch.arange(batch_size).view(-1, 1, 1).expand(-1, seq_len, k)
        seq_idx = torch.arange(seq_len).view(1, -1, 1).expand(batch_size, -1, k)
        correct[batch_idx, seq_idx, self.top_ids] = 1.0 / k

        return correct

    @property
    def loss_sum(self):
        loss = self.kldiv_loss()
        if self.prev:
            loss += self.prev.loss_sum

        return loss
