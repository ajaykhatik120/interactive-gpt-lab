import torch

from tokenizer.word_tokenizer import WordTokenizer
from model.embeddings import GPTInputEmbedding
from config import config

corpus = """
The cat drinks milk.
The dog eats food.
The cat likes food.
The dog drinks water.
"""

# Build vocabulary
tokenizer = WordTokenizer()
tokenizer.build_vocabulary(corpus)

# Encode two sequences
sentence_1 = tokenizer.encode("The cat drinks milk")
sentence_2 = tokenizer.encode("The dog eats food")

input_ids = torch.tensor([sentence_1, sentence_2], dtype=torch.long)

print("Input IDs:")
print(input_ids)

print("\nInput shape:")
print(input_ids.shape)


# Create GPT input embedding
embedding_layer = GPTInputEmbedding(
    vocab_size=tokenizer.vocab_size,
    block_size=config.block_size,
    embedding_dim=config.n_embd,
)


# Forward pass
output = embedding_layer(input_ids)

print("\nFinal embedding output:")
print(output)

print("\nFinal output shape:")
print(output.shape)


# Inspect token and position embeddings separately
token_vectors = embedding_layer.token_embedding(input_ids)

position_vectors = embedding_layer.position_embedding(input_ids)

print("\nToken embedding shape:")
print(token_vectors.shape)

print("\nPosition embedding shape:")
print(position_vectors.shape)

print("\nFirst token vector:")
print(token_vectors[0, 0])

print("\nFirst position vector:")
print(position_vectors[0, 0])

print("\nCombined vector:")
print(output[0, 0])
