import torch.nn as nn

from model.attention import MultiHeadCausalSelfAttention
from model.feed_forward import FeedForward
from model.normalization import LayerNormalization


class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm Transformer block:
        x1 = x + Attention(LN(x))
        x2 = x1 + MLP(LN(x1))
    """

    def __init__(self, embedding_dim, num_heads, block_size):
        super().__init__()
        self.ln1 = LayerNormalization(embedding_dim)
        self.ln2 = LayerNormalization(embedding_dim)
        self.attention = MultiHeadCausalSelfAttention(
            embedding_dim, num_heads, block_size
        )
        self.feed_forward = FeedForward(embedding_dim)

    def forward(self, x, return_debug=False):
        if not return_debug:
            attention_output, attention_weights = self.attention(self.ln1(x))
            x = x + attention_output
            mlp_output = self.feed_forward(self.ln2(x))
            x = x + mlp_output
            return x, attention_weights

        residual_input = x
        normalized_for_attention = self.ln1(x)

        attention_output, attention_weights, attention_debug = self.attention(
            normalized_for_attention, return_debug=True
        )

        after_attention_residual = residual_input + attention_output

        normalized_for_mlp = self.ln2(after_attention_residual)
        mlp_output, mlp_debug = self.feed_forward(normalized_for_mlp, return_debug=True)

        block_output = after_attention_residual + mlp_output

        debug = {
            "residual_input": residual_input,
            "normalized_for_attention": normalized_for_attention,
            "attention": attention_debug,
            "attention_output": attention_output,
            "after_attention_residual": after_attention_residual,
            "normalized_for_mlp": normalized_for_mlp,
            "mlp": mlp_debug,
            "block_output": block_output,
        }

        return block_output, attention_weights, debug
