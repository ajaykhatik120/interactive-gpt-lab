import torch

from tokenizer.word_tokenizer import WordTokenizer
from model.embeddings import GPTInputEmbedding
from model.attention import CausalSelfAttention
from config import config

corpus = """
The cat drinks milk.
The dog eats food.
The cat likes food.
The dog drinks water.
"""

# 1. Build tokenizer
tokenizer = WordTokenizer()
tokenizer.build_vocabulary(corpus)

# 2. Create a sample input
tokens = tokenizer.encode("The cat drinks milk")

input_ids = torch.tensor([tokens], dtype=torch.long)

# 3. Create token + position embeddings
embedding_layer = GPTInputEmbedding(
    vocab_size=tokenizer.vocab_size,
    block_size=config.block_size,
    embedding_dim=config.n_embd,
)

x = embedding_layer(input_ids)

print("Input embedding shape:", x.shape)

# 4. Create single-head attention
attention = CausalSelfAttention(
    embedding_dim=config.n_embd, head_size=config.n_embd, block_size=config.block_size
)

# 5. Forward pass
output, weights = attention(x)

print("\nAttention output shape:")
print(output.shape)

print("\nAttention weights shape:")
print(weights.shape)

print("\nAttention weights:")
print(weights)

# 6. Verify causal masking
print("\nFuture attention weights:")

for i in range(weights.shape[1]):
    for j in range(i + 1, weights.shape[2]):
        print(f"Position {i} -> Position {j}:", weights[0, i, j].item())

# 7. Check that each row sums to 1
print("\nAttention row sums:")
print(weights.sum(dim=-1))
