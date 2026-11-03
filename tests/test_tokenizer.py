
from tokenizer.word_tokenizer import WordTokenizer


corpus = """
The cat drinks milk.
The dog eats food.
The cat likes food.
"""

tokenizer = WordTokenizer()

# Build vocabulary
tokenizer.build_vocabulary(corpus)

print("Vocabulary size:", tokenizer.vocab_size)

print("\nVocabulary:")
print(tokenizer.stoi)

# Encode
sentence = "The cat drinks milk."

encoded = tokenizer.encode(sentence)

print("\nOriginal:", sentence)
print("Encoded:", encoded)

# Decode
decoded = tokenizer.decode(encoded)

print("\nDecoded:", decoded)

# Unknown word test
unknown_sentence = "The elephant drinks milk."

print("\nUnknown word test:")
print(tokenizer.encode(unknown_sentence))


