from typing import List

import torch


# The class represents a description of all computations in HMoE for a given input.
# The input may contain multiple batches. It is assumed that this class will be used
# during model evaluation, not during training.
class Scenario:
    attn_mask: torch.Tensor
    chains: List[torch.Tensor]  # [level, (batch_size, position, top-k in tensor)] - top-k indices
    weights: List[List[torch.Tensor]]  # [level, (batch_size, position, top-k in tensor)] - top-k weights
    latents: List[List[torch.Tensor]]  # [level, position, (batch_size in tensor)] - inp + activations

    def __init__(self, attn_mask, chains, weights, latents):
        self.attn_mask = attn_mask
        self.chains = chains
        self.weights = weights
        self.latents = latents

    def append_level(self, scenario):
        self.chains += scenario.chains
        self.weights += scenario.weights
        self.latents += scenario.latents

    def prepend_level(self, scenario):
        self.chains = scenario.chains + self.chains
        self.weights = scenario.weights + self.weights
        self.latents = scenario.latents + self.latents

    def clone(self):
        return Scenario(
            attn_mask=self.attn_mask,
            chains=[c.clone() for c in self.chains],
            weights=[[w.clone() for w in lvl] for lvl in self.weights],
            latents=[[l.clone() for l in lvl] for lvl in self.latents],
        )

    def __len__(self):
        return len(self.chains)

    def __getitem__(self, idx):
        """
        Scenario[i] -> Scenario with one level
        Scenario[i:j] -> Scenario with many levels
        """
        if isinstance(idx, slice):
            return Scenario(
                attn_mask=self.attn_mask,
                chains=self.chains[idx],
                weights=self.weights[idx],
                latents=self.latents[idx],
            )

        if isinstance(idx, int):
            return Scenario(
                attn_mask=self.attn_mask,
                chains=[self.chains[idx]],
                weights=[self.weights[idx]],
                latents=[self.latents[idx]],
            )

        raise TypeError(f"Invalid index type: {type(idx)}")
