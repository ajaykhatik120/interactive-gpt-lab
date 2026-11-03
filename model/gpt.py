import torch
import torch.nn as nn

from model.embeddings import GPTInputEmbedding
from model.transformer_block import TransformerBlock
from model.normalization import LayerNormalization


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, block_size):
        super().__init__()

        self.block_size = block_size
        self.vocab_size = vocab_size

        self.embedding = GPTInputEmbedding(vocab_size, block_size, embedding_dim)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embedding_dim, num_heads, block_size)
                for _ in range(num_layers)
            ]
        )

        self.final_norm = LayerNormalization(embedding_dim)
        self.language_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, token_ids, return_debug=False):
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, sequence).")

        batch_size, sequence_length = token_ids.shape

        if sequence_length > self.block_size:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"block_size {self.block_size}."
            )

        if return_debug:
            token_vectors = self.embedding.token_embedding(token_ids)
            position_vectors = self.embedding.position_embedding(token_ids)
            x = token_vectors + position_vectors.unsqueeze(0)

            embedding_debug = {
                "token_vectors": token_vectors,
                "position_vectors": position_vectors,
                "combined_embeddings": x,
            }

            all_attention_weights = []
            all_block_debug = []

            for block in self.blocks:
                x, attention_weights, block_debug = block(x, return_debug=True)
                all_attention_weights.append(attention_weights)
                all_block_debug.append(block_debug)

            before_final_norm = x
            x = self.final_norm(x)
            logits = self.language_head(x)

            debug = {
                "embeddings": embedding_debug,
                "blocks": all_block_debug,
                "before_final_norm": before_final_norm,
                "after_final_norm": x,
                "logits": logits,
            }

            return logits, all_attention_weights, debug

        x = self.embedding(token_ids)
        all_attention_weights = []

        for block in self.blocks:
            x, attention_weights = block(x)
            all_attention_weights.append(attention_weights)

        x = self.final_norm(x)
        logits = self.language_head(x)

        return logits, all_attention_weights
