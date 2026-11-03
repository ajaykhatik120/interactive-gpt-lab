import torch

from tokenizer.word_tokenizer import WordTokenizer
from model.embeddings import GPTInputEmbedding
from model.transformer_block import TransformerBlock
from config import config

corpus = """
The cat drinks milk.
The dog eats food.
The cat likes food.
The dog drinks water.
my name is ajay.
"""

tokenizer = WordTokenizer()
tokenizer.build_vocabulary(corpus)

tokens = tokenizer.encode("The cat drinks milk")

input_ids = torch.tensor([tokens], dtype=torch.long)

# Input embeddings
embedding_layer = GPTInputEmbedding(
    vocab_size=tokenizer.vocab_size,
    block_size=config.block_size,
    embedding_dim=config.n_embd,
)

x = embedding_layer(input_ids)

print("Input shape:", x.shape)

# Transformer block
block = TransformerBlock(
    embedding_dim=config.n_embd, num_heads=config.n_head, block_size=config.block_size
)

output, attention_weights = block(x)

print("\nTransformer output shape:")
print(output.shape)

print("\nAttention weights shape:")
print(attention_weights.shape)

print("\nOutput first token:")
print(output[0, 0])
