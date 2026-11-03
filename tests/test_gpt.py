import torch

from tokenizer.word_tokenizer import WordTokenizer
from model.gpt import GPTLanguageModel
from config import config

corpus = """
The cat drinks milk.
The dog eats food.
The cat likes food.
The dog drinks water.
"""

# 1. Build vocabulary
tokenizer = WordTokenizer()
tokenizer.build_vocabulary(corpus)

print("Vocabulary size:", tokenizer.vocab_size)


# 2. Encode input
tokens = tokenizer.encode("The cat drinks milk")

input_ids = torch.tensor([tokens], dtype=torch.long)

print("\nInput IDs:", input_ids)
print("Input shape:", input_ids.shape)


# 3. Create complete GPT
model = GPTLanguageModel(
    vocab_size=tokenizer.vocab_size,
    embedding_dim=config.n_embd,
    num_heads=config.n_head,
    num_layers=config.n_layer,
    block_size=config.block_size,
)


# 4. Forward pass
logits, attention_weights = model(input_ids)

print("\nLogits shape:")
print(logits.shape)

print("\nNumber of Transformer blocks:")
print(len(attention_weights))

print("\nAttention shape of first block:")
print(attention_weights[0].shape)


# 5. Convert last-position logits to probabilities
last_token_logits = logits[:, -1, :]

probabilities = torch.softmax(last_token_logits, dim=-1)

print("\nProbability shape:")
print(probabilities.shape)

print("\nProbability sum:")
print(probabilities.sum(dim=-1))


# 6. Find the most probable token
predicted_id = probabilities.argmax(dim=-1).item()

print("\nPredicted next token:")
print(tokenizer.decode([predicted_id]))

# Display all token probabilities
print("\nVocabulary probabilities:")

for token_id, probability in enumerate(probabilities[0]):

    token = tokenizer.itos[token_id]

    print(f"{token:10s}: {probability.item():.4f}")
