import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, token_ids):
        return self.embedding(token_ids)


class PositionalEmbedding(nn.Module):
    def __init__(self, block_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(block_size, embedding_dim)

    def forward(self, token_ids):
        sequence_length = token_ids.size(1)
        positions = torch.arange(sequence_length, device=token_ids.device)
        return self.embedding(positions)


class GPTInputEmbedding(nn.Module):
    def __init__(self, vocab_size, block_size, embedding_dim):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embedding_dim)
        self.position_embedding = PositionalEmbedding(block_size, embedding_dim)

    def forward(self, token_ids):
        token_vectors = self.token_embedding(token_ids)
        position_vectors = self.position_embedding(token_ids)
        return token_vectors + position_vectors.unsqueeze(0)
