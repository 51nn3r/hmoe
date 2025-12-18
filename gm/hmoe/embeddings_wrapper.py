import torch
from torch import nn


class EmbeddingsWrapper(nn.Module):
    def __init__(self, model, vocab_size, d_model):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional embeddings
        self.pos_embedding = nn.Embedding(1000, d_model)  # maximum of 1000 positions

        self.model = model
        # Your HMoE model

        # Output head
        self.output_head = nn.Linear(d_model, vocab_size)

        self.depth = d_model
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_length = input_ids.shape

        # Token embeddings
        token_embeddings = self.embedding(input_ids)  # [batch, seq, depth]

        # Positional embeddings
        positions = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        pos_embeddings = self.pos_embedding(positions)  # [1, seq, depth]

        # Sum embeddings
        embeddings = token_embeddings + pos_embeddings

        # HMoE forward pass
        res = self.model(embeddings, attention_mask)  # [batch, seq, depth]

        # Logits for the next token
        res['out'] = self.output_head(res['out'])  # [batch, seq, vocab_size]

        return res
