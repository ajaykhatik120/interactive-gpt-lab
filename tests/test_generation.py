import torch

from tokenizer.word_tokenizer import WordTokenizer
from model.gpt import GPTLanguageModel
from inference.predictor import GPTPredictor
from config import config

corpus = """
The cat drinks milk.
The dog eats food.
The cat likes food.
The dog drinks water.
"""

# 1. Rebuild the same vocabulary
tokenizer = WordTokenizer()
tokenizer.build_vocabulary(corpus)

# 2. Recreate the model architecture
model = GPTLanguageModel(
    vocab_size=tokenizer.vocab_size,
    embedding_dim=config.n_embd,
    num_heads=config.n_head,
    num_layers=config.n_layer,
    block_size=config.block_size,
)

# 3. Load trained weights
model.load_state_dict(torch.load("checkpoints/gpt_model.pth", map_location="cpu"))

model.eval()

print("Trained model loaded successfully!")

# 4. Create predictor
predictor = GPTPredictor(model=model, tokenizer=tokenizer)

# 5. Generate text using greedy decoding
prompt = "The cat"

generated_text = predictor.generate(prompt=prompt, max_new_tokens=15, greedy=True)

print("\nPrompt:")
print(prompt)

print("\nGenerated text:")
print(generated_text)

# 6. Try sampling
sampled_text = predictor.generate(
    prompt=prompt, max_new_tokens=15, temperature=0.8, top_k=5, greedy=False
)

print("\nSampled generation:")
print(sampled_text)
