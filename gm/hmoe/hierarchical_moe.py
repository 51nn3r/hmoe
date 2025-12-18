from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from gm.hmoe.deep_dark_gate import DeepDarkGate
from gm.hmoe.dummy_expert import DummyExpert
from gm.hmoe.experts_storage import ExpertsStorage
from gm.hmoe.hmoe_component import HMoeComponent
from gm.hmoe.hmoe_usage import HMoeUsage
from gm.hmoe.transformer_expert import TransformerExpert


class HierarchicalMoE(HMoeComponent):
    '''Hierarchical mixture of experts'''

    def __init__(
            self,
            experts_count,
            chain_size,
            gate: nn.Module,
            experts_storage: ExpertsStorage,
            top_k=1,
            tau=0.1,
            num_heads=2,
            d_model=None,
            key_dim=None,
            dropout=0.,
    ):
        super().__init__(gate=gate, experts_storage=experts_storage, chain_size=chain_size, d_model=d_model)

        self.experts_count = experts_count + 1  # n experts per position + 1 no-op expert
        self.top_k = top_k
        self.tau = tau
        self.num_heads = num_heads
        self.d_model = d_model
        self.key_dim = key_dim
        self.dropout = dropout
        self.gate_out_projection = nn.Linear(d_model, self.experts_count)
        experts_usage_stat = torch.zeros(self.experts_count, dtype=torch.int32)
        self.register_buffer("experts_usage_stat", experts_usage_stat)

        # Trainable vectors used by the gate — model parameters
        self.trainable_vectors = None

    def _build(self, x):
        if self.d_model is None:
            self.d_model = x.shape[-1]

        if self.key_dim is None:
            self.key_dim = self.d_model

        if self.prev_hmoe_chian_size is not None:
            self.trainable_vectors = nn.Parameter(torch.randn(self.prev_hmoe_chian_size, self.d_model, device=x.device))
            nn.init.xavier_uniform_(self.trainable_vectors)

    def _gumbel_softmax(self, logits, tau=1.0, hard=False, dim=-1, use_l2=False):
        '''Gumbel-Softmax with top-k and L2 normalization (sum of squares = 1)'''
        # Gumbel(0,1) noise
        gumbels = -torch.empty_like(logits).exponential_().log()
        scores = (logits + gumbels) / tau

        # top-k over stochastic scores
        topk = scores.topk(self.top_k, dim=dim)
        index = topk.indices
        mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(dim, index, True)

        # softmax only within top-k, outside = 0
        masked = scores.masked_fill(~mask, float('-inf'))
        y_soft = F.softmax(masked, dim=dim)

        if use_l2:
            # L2 normalization over dim: sum of squares = 1
            eps = torch.finfo(y_soft.dtype).eps
            l2 = torch.linalg.vector_norm(y_soft, ord=2, dim=dim, keepdim=True).clamp_min(eps)
            y_soft = y_soft / l2

        if hard:
            # forward: same values only on top-k; backward: gradients as in y_soft
            y_hard = torch.zeros_like(scores).masked_scatter(mask, y_soft.masked_select(mask))
            return y_hard - y_soft.detach() + y_soft, scores

        return y_soft, scores

    def _select_experts(self, top_weights, top_indices, x, attn_mask=None):
        '''Select and execute experts for a specific position in the chain'''
        batch_size = x.size(0)
        device = x.device

        # Accumulate:
        #  - total weight per expert for each batch element
        #  - how many times an expert appears in top-K for each b
        expert_usage = torch.zeros(
            batch_size, self.experts_count,
            device=device, dtype=torch.int32
        )  # (B, E)

        weight_per_expert = torch.zeros(
            batch_size, self.experts_count,
            device=device, dtype=top_weights.dtype
        )  # (B, E)

        # total weights
        weight_per_expert.scatter_add_(1, top_indices, top_weights)

        # hit counts
        ones = torch.ones_like(top_indices, dtype=expert_usage.dtype)  # (B, K)
        expert_usage.scatter_add_(1, top_indices, ones)

        # Output
        output = torch.zeros_like(x)

        # List of actually used experts
        used_experts = (expert_usage.sum(dim=0) != 0).nonzero(as_tuple=True)[0]  # (E_used,)

        # Group by expert and process in a single pass per expert
        for expert_idx in used_experts.tolist():
            # batches where this expert has non-zero weight
            w = weight_per_expert[:, expert_idx]  # (B,)
            idx = (w != 0).nonzero(as_tuple=True)[0]  # (M,)
            if idx.numel() == 0:
                continue

            expert = self.experts_storage.experts[expert_idx]

            current_mask = attn_mask
            if attn_mask is not None:
                current_mask = current_mask[idx] if current_mask.size(0) == x.size(0) else current_mask

            y = expert(x[idx], current_mask)  # (M, T, D)

            # Weighted aggregation into output
            output[idx] += y * w[idx].unsqueeze(-1).unsqueeze(-1)

        return output, expert_usage

    def forward(self, x, attn_mask=None):
        super().forward(x)

        # Get weights from the gate
        res = self.gate(x, attn_mask=attn_mask)  # [batch_size, bottom_level_m, n]
        weights, gates_output, prev_exp_usage = res.values()  # [batch_size, bottom_level_m, n]

        x, attn_mask = self._extend_x_and_mask_with_trainable_vectors(x, attn_mask)

        # Zero-chain processing
        computation = x

        # Project gate output to the required expert dimensionality
        chain_weights = self.gate_out_projection(weights)
        # Gumbel-Softmax
        gate_out, gumbel_scores = self._gumbel_softmax(chain_weights, tau=self.tau, hard=True)
        # Top-k expert selection
        top_weights, top_indices = torch.topk(gate_out, self.top_k, dim=-1)  # (B, T, K)

        vals = torch.ones_like(top_indices, dtype=self.experts_usage_stat.dtype)
        # self.experts_usage_stat.index_add_(0, top_indices.view(-1), vals.view(-1))

        batch_size = x.size(0)
        end_mask = torch.ones(batch_size, dtype=torch.bool, device=x.device)
        for chain_idx in range(self.chain_size):
            self.experts_usage_stat.index_add_(0, top_indices[end_mask, chain_idx].view(-1),
                                               vals[end_mask, chain_idx].view(-1))
            current_weights = top_weights[:, chain_idx]
            current_indices = top_indices[:, chain_idx]

            if x.size(0) == attn_mask.size(0):
                current_mask = attn_mask[end_mask]
            else:
                current_mask = attn_mask

            y, usage = self._select_experts(
                current_weights[end_mask],
                current_indices[end_mask],
                computation[end_mask],
                current_mask
            )
            # computation[end_mask] += y
            computation[end_mask] = y

            is_first_expert_zero = current_indices[:, 0] == 0
            is_first_weight_dominant = current_weights[:, 0] > 2 * current_weights[:, 1:].sum(dim=1)

            should_stop = is_first_expert_zero & is_first_weight_dominant
            end_mask = end_mask & ~should_stop

            if chain_idx > 0 and not torch.any(end_mask):
                break

        if self.trainable_vectors is not None:
            computation = computation[:, :self.trainable_vectors.size(0), :]
            gates_output = torch.cat(
                [gates_output, torch.sum(computation[:, None, :, 1:], dim=-2)],
                axis=-3
            )

        return {
            'out': computation,
            'gates_out': gates_output,
            'exp_usage': HMoeUsage(gumbel_scores, top_indices, prev_exp_usage),
        }

    @staticmethod
    def create_hierarchical_moe(
            experts_count,
            chain_sizes: List[int],
            top_k: int = 1,
            tau: float = 0.1,
            num_heads: int = 2,
            d_model: int = 32,
            dim_feedforward: int = 128,
            dropout: float = 0.1,
    ):
        experts_storage = ExpertsStorage(experts_count)
        experts = nn.ModuleList([DummyExpert(mode='identity')])
        for idx, _ in enumerate(range(experts_count)):
            if idx % 1000 == 0:
                print(f'[*] experts init: {idx} / {experts_count}')

            experts.append(TransformerExpert(
                d_model,
                num_heads,
                dim_feedforward,
                dropout,
            ))

        experts_storage.set_experts(experts)
        print(f'[*] storage params count: {sum(p.numel() for p in experts_storage.parameters())}')

        # Create the bottom-level gate
        fd_h = hierarchical = DeepDarkGate(d_model=d_model, num_heads=4, dropout_rate=0.1)

        for chain_size in chain_sizes:
            hierarchical = HierarchicalMoE(
                experts_count,
                chain_size=chain_size,
                gate=hierarchical,
                experts_storage=experts_storage,
                top_k=top_k,
                tau=tau,
                num_heads=num_heads,
                d_model=d_model,
                dropout=dropout,
            )
            fd_h.set_prev_hmoe(hierarchical)
            fd_h = hierarchical

        return hierarchical, experts_storage
