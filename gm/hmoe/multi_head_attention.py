import math

import torch
import torch.nn.functional as F
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=None, key_dim=None, num_heads=8, dropout=0.0, use_bias=True):
        super().__init__()

        self.d_model = d_model
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.use_bias = use_bias
        self._built = False

    def _build(self, x):
        if self._built:
            return

        if self.d_model is None:
            self.d_model = x.shape[-1]

        if self.key_dim is None:
            self.key_dim = self.d_model

        self.W_q = nn.Parameter(torch.empty(self.d_model, self.num_heads, self.key_dim, device=x.device))
        self.W_k = nn.Parameter(torch.empty(self.d_model, self.num_heads, self.key_dim, device=x.device))
        self.W_v = nn.Parameter(torch.empty(self.d_model, self.num_heads, self.key_dim, device=x.device))
        self.b_q = nn.Parameter(torch.zeros(self.num_heads, self.key_dim, device=x.device)) if self.use_bias else None
        self.b_k = nn.Parameter(torch.zeros(self.num_heads, self.key_dim, device=x.device)) if self.use_bias else None
        self.b_v = nn.Parameter(torch.zeros(self.num_heads, self.key_dim, device=x.device)) if self.use_bias else None

        self.W_o = nn.Parameter(torch.empty(self.num_heads, self.key_dim, self.d_model, device=x.device))
        self.b_o = nn.Parameter(torch.zeros(self.d_model, device=x.device)) if self.use_bias else None

        self.attn_dropout = nn.Dropout(self.dropout_rate)
        self.proj_dropout = nn.Dropout(self.dropout_rate)

        self.reset_parameters()
        self._built = True

    def reset_parameters(self):
        for W in (self.W_q, self.W_k, self.W_v, self.W_o):
            nn.init.normal_(W, mean=0.0, std=0.02)

        for b in (self.b_q, self.b_k, self.b_v, self.b_o):
            if b is not None:
                nn.init.zeros_(b)

    def forward(
            self,
            q,
            k=None,
            v=None,
            attn_mask=None,
            key_padding_mask=None,
            need_weights=False,
    ):
        self._build(q)

        if k is None: k = q
        if v is None: v = k

        batch_size, time_steps, depth = q.shape
        key_time_steps = k.size(1)
        scale = 1.0 / math.sqrt(self.key_dim)

        Q = torch.einsum('bsd,dhk->bshk', q, self.W_q)
        K = torch.einsum('bsd,dhk->bshk', k, self.W_k)
        V = torch.einsum('bsd,dhk->bshk', v, self.W_v)
        if self.b_q is not None:
            Q = Q + self.b_q
            K = K + self.b_k
            V = V + self.b_v

        attn_logits = torch.einsum('btnk,bfnk->bntf', Q, K) * scale

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_logits = attn_logits.masked_fill(attn_mask, float('-inf'))
            else:
                attn_logits = attn_logits + attn_mask

        if key_padding_mask is not None:
            m = key_padding_mask.view(batch_size, 1, 1, key_time_steps)
            attn_logits = attn_logits.masked_fill(m, float('-inf'))

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.einsum('bntf,bfnk->btnk', attn_weights, V)
        out = torch.einsum('btnk,nkd->btd', context, self.W_o)
        if self.b_o is not None:
            out = out + self.b_o

        out = self.proj_dropout(out)

        if need_weights:
            return out, attn_weights.mean(dim=1)

        return out
