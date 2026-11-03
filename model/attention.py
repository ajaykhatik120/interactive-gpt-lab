import math
import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    """
    One attention head. return_debug=True exposes the real intermediate tensors.
    """

    def __init__(self, embedding_dim, head_size, block_size):
        super().__init__()
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)

        causal_mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("causal_mask", causal_mask)

        self.head_size = head_size

    def forward(self, x, return_debug=False):
        batch_size, sequence_length, _ = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        raw_scores = q @ k.transpose(-2, -1)
        raw_scores = raw_scores / math.sqrt(self.head_size)

        allowed_positions = self.causal_mask[:sequence_length, :sequence_length]
        masked_scores = raw_scores.masked_fill(
            allowed_positions == 0,
            float("-inf")
        )

        attention_weights = torch.softmax(masked_scores, dim=-1)
        output = attention_weights @ v

        if not return_debug:
            return output, attention_weights

        debug = {
            "query": q,
            "key": k,
            "value": v,
            "raw_scores": raw_scores,
            "masked_scores": masked_scores,
            "attention_weights": attention_weights,
            "weighted_values": output,
            "causal_mask": allowed_positions
        }
        return output, attention_weights, debug


class MultiHeadCausalSelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, block_size):
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads.")

        self.num_heads = num_heads
        self.head_size = embedding_dim // num_heads

        self.heads = nn.ModuleList([
            CausalSelfAttention(
                embedding_dim=embedding_dim,
                head_size=self.head_size,
                block_size=block_size
            )
            for _ in range(num_heads)
        ])

        self.output_projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x, return_debug=False):
        if return_debug:
            head_outputs = []
            head_weights = []
            head_debug = []

            for head in self.heads:
                output, weights, debug = head(x, return_debug=True)
                head_outputs.append(output)
                head_weights.append(weights)
                head_debug.append(debug)

            concatenated = torch.cat(head_outputs, dim=-1)
            projected = self.output_projection(concatenated)

            debug = {
                "heads": head_debug,
                "head_outputs": torch.stack(head_outputs, dim=1),
                "concatenated_heads": concatenated,
                "output_projection": projected
            }

            return projected, torch.stack(head_weights, dim=1), debug

        head_outputs = []
        head_weights = []

        for head in self.heads:
            output, weights = head(x)
            head_outputs.append(output)
            head_weights.append(weights)

        concatenated = torch.cat(head_outputs, dim=-1)
        projected = self.output_projection(concatenated)

        return projected, torch.stack(head_weights, dim=1)
