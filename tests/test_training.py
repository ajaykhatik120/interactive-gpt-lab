import torch

from tokenizer.word_tokenizer import WordTokenizer
from dataset.dataset import GPTDataset
from model.gpt import GPTLanguageModel
from training.trainer import GPTTrainer
from config import config

# Training corpus
corpus = """
The cat drinks milk.
The dog eats food.
The cat likes food.
The dog drinks water.
"""

# 1. Build vocabulary
tokenizer = WordTokenizer()
tokenizer.build_vocabulary(corpus)

# 2. Encode the complete corpus with EOS markers
token_ids = tokenizer.encode_corpus(corpus)

# 3. Create training dataset
dataset = GPTDataset(token_ids=token_ids, block_size=config.block_size)

print("Vocabulary size:", tokenizer.vocab_size)
print("Total tokens:", len(token_ids))
print("Training examples:", len(dataset))

# 4. Create GPT model
model = GPTLanguageModel(
    vocab_size=tokenizer.vocab_size,
    embedding_dim=config.n_embd,
    num_heads=config.n_head,
    num_layers=config.n_layer,
    block_size=config.block_size,
)

# 5. Create trainer
trainer = GPTTrainer(model=model, learning_rate=config.learning_rate)

# 6. Train
trained_model = trainer.train(dataset=dataset, epochs=500, batch_size=config.batch_size)

# 7. Save trained model parameters
torch.save(trained_model.state_dict(), "checkpoints/gpt_model.pth")

print("\nModel saved successfully!")
