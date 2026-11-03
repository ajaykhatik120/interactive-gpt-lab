import torch

from tokenizer.word_tokenizer import WordTokenizer
from model.embeddings import GPTInputEmbedding
from model.attention import MultiHeadCausalSelfAttention
from config import config

corpus = """
The cat drinks milk.
The dog eats food.
The cat likes food.
The dog drinks water.
"""

tokenizer = WordTokenizer()
tokenizer.build_vocabulary(corpus)

tokens = tokenizer.encode("The cat drinks milk")

input_ids = torch.tensor([tokens], dtype=torch.long)

# Create input embeddings
embedding_layer = GPTInputEmbedding(
    vocab_size=tokenizer.vocab_size,
    block_size=config.block_size,
    embedding_dim=config.n_embd,
)

x = embedding_layer(input_ids)

# Create multi-head attention
attention = MultiHeadCausalSelfAttention(
    embedding_dim=config.n_embd, num_heads=config.n_head, block_size=config.block_size
)

# Forward pass
output, weights = attention(x)

print("Input shape:", x.shape)

print("\nNumber of heads:", config.n_head)

print("\nOutput shape:", output.shape)

print("\nAttention weights shape:", weights.shape)

# Check each head's attention distribution
print("\nAttention row sums for every head:")
print(weights.sum(dim=-1))

# Check causal masking for every head
future_weights = torch.triu(weights, diagonal=1)

print("\nMaximum future attention weight:")
print(future_weights.max().item())
