from tokenizer.word_tokenizer import WordTokenizer
from dataset.dataset import GPTDataset


corpus = """
The cat drinks milk.
The dog eats food.
The cat likes food.
The dog drinks water.
"""

# Build tokenizer
tokenizer = WordTokenizer()
tokenizer.build_vocabulary(corpus)

# Encode entire corpus with sentence boundaries
token_ids = tokenizer.encode_corpus(corpus)

print("Encoded corpus:")
print(token_ids)

# Create dataset
dataset = GPTDataset(
    token_ids=token_ids,
    block_size=3
)

print("\nDataset size:", len(dataset))

# Inspect first training example
x, y = dataset[0]

print("\nInput IDs:", x.tolist())
print("Target IDs:", y.tolist())

print("\nInput text:", tokenizer.decode(x.tolist()))
print("Target text:", tokenizer.decode(y.tolist()))