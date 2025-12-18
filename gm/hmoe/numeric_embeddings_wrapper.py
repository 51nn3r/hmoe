import torch
from torch import nn


class NumericEmbeddingsWrapper(nn.Module):
    """
    Wrapper for attention / HMoE models applied to financial time series (seq2seq).

    Input:
        x: [B, L, D_in]  -- numeric features (already in the form of deltas / log-returns / ratios)
        attention_mask: as expected by the inner model (usually [B, 1, L, L])

    Output:
        res: dict, as returned by self.model, but:
            res["out"] : [B, L, D_in] -- predictions in the NORMALIZED feature space
            (optional) res["out_denorm"] : [B, L, D_in] -- in the “real” feature scale
    """

    def __init__(
            self,
            model: nn.Module,
            input_dim: int,
            d_model: int,
            max_len: int = 1000,
            use_learnable_norm: bool = False,
            eps: float = 1e-6,
    ):
        super().__init__()

        self.model = model
        self.input_dim = input_dim
        self.d_model = d_model
        self.eps = eps

        # Projection from numeric features → d_model
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional embeddings
        self.pos_embedding = nn.Embedding(max_len, d_model)

        # Static feature normalization (mean/std provided externally)
        self.register_buffer("feat_mean", torch.zeros(input_dim))
        self.register_buffer("feat_std", torch.ones(input_dim))

        # Additional learnable normalization (optional)
        self.use_learnable_norm = use_learnable_norm
        if use_learnable_norm:
            self.norm_scale = nn.Parameter(torch.ones(input_dim))
            self.norm_bias = nn.Parameter(torch.zeros(input_dim))
        else:
            self.norm_scale = None
            self.norm_bias = None

        # Output head: predict the same features (seq2seq)
        self.output_head = nn.Linear(d_model, input_dim)

    # ---------- normalization / denormalization ----------

    @torch.no_grad()
    def set_normalization_stats(self, mean: torch.Tensor, std: torch.Tensor):
        """
        Set normalization statistics (mean/std) from the training dataset.

        mean, std: [input_dim]
        """
        mean = mean.to(self.feat_mean.device)
        std = std.to(self.feat_std.device).clamp_min(self.eps)

        self.feat_mean.copy_(mean)
        self.feat_std.copy_(std)

    def normalize_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Feature normalization (for both inputs and targets).

        x: [..., D_in]
        """
        # Broadcast mean/std to [..., D_in]
        mean = self.feat_mean.view(*(1 for _ in range(x.dim() - 1)), -1)
        std = self.feat_std.view(*(1 for _ in range(x.dim() - 1)), -1).clamp_min(self.eps)

        x_norm = (x - mean) / std

        if self.use_learnable_norm:
            scale = self.norm_scale.view(*(1 for _ in range(x.dim() - 1)), -1)
            bias = self.norm_bias.view(*(1 for _ in range(x.dim() - 1)), -1)
            x_norm = x_norm * scale + bias

        return x_norm

    def denormalize_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inverse transformation of normalize_features (for predictions).

        x: [..., D_in] in the NORMALIZED space.
        """
        x_den = x

        if self.use_learnable_norm:
            scale = self.norm_scale.view(*(1 for _ in range(x.dim() - 1)), -1)
            bias = self.norm_bias.view(*(1 for _ in range(x.dim() - 1)), -1)
            x_den = (x_den - bias) / (scale + self.eps)

        mean = self.feat_mean.view(*(1 for _ in range(x.dim() - 1)), -1)
        std = self.feat_std.view(*(1 for _ in range(x.dim() - 1)), -1)
        x_den = x_den * std + mean

        return x_den

    # ---------- forward ----------

    def forward(
            self,
            x: torch.Tensor,  # [B, L, D_in] — engineered features
            attention_mask: torch.Tensor | None = None,
            denorm: bool = False,
    ):
        B, L, D = x.shape
        device = x.device

        # 1) input normalization
        x_norm = self.normalize_features(x)  # [B, L, D_in]

        # 2) projection to d_model
        h = self.input_proj(x_norm)  # [B, L, d_model]

        # 3) positional embeddings
        positions = torch.arange(L, device=device).unsqueeze(0)  # [1, L]
        pos_emb = self.pos_embedding(positions)  # [1, L, d_model]

        h = h + pos_emb  # [B, L, d_model]

        # 4) internal attention / HMoE seq2seq model
        # expected that self.model returns a dict with key "out": [B, L, d_model]
        res = self.model(h, attention_mask)  # res["out"]: [B, L, d_model]

        # 5) output head in feature space (NORMALIZED)
        pred_norm = self.output_head(res["out"])  # [B, L, D_in]
        res["out"] = pred_norm

        # 6) optional — predictions in the “real” feature scale
        if denorm:
            res["out_denorm"] = self.denormalize_features(pred_norm)  # [B, L, D_in]

        return res
