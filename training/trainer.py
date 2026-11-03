import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from model.gpt import GPTLanguageModel
from dataset.dataset import GPTDataset
from config import config


class GPTTrainer:

    def __init__(self, model, learning_rate):

        self.model = model

        # Cross entropy compares raw logits with correct token IDs
        self.loss_function = nn.CrossEntropyLoss()

        # Adam optimizer updates model parameters
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    def train(self, dataset, epochs, batch_size):

        # Do not request batches larger than our dataset
        effective_batch_size = min(batch_size, len(dataset))

        loader = DataLoader(
            dataset, batch_size=effective_batch_size, shuffle=True, drop_last=False
        )

        self.model.train()

        for epoch in range(epochs):

            total_loss = 0.0
            batches = 0

            for x, y in loader:

                # 1. Clear old gradients
                self.optimizer.zero_grad()

                # 2. Forward pass
                logits, _ = self.model(x)

                # 3. Flatten token positions for cross entropy
                batch_size, sequence_length, vocab_size = logits.shape

                logits = logits.reshape(batch_size * sequence_length, vocab_size)

                targets = y.reshape(batch_size * sequence_length)

                # 4. Calculate prediction loss
                loss = self.loss_function(logits, targets)

                # 5. Backpropagation
                loss.backward()

                # 6. Update model parameters
                self.optimizer.step()

                total_loss += loss.item()
                batches += 1

            average_loss = total_loss / batches

            if (epoch + 1) % 25 == 0 or epoch == 0:

                print(f"Epoch [{epoch + 1}/{epochs}] " f"Loss: {average_loss:.4f}")

        print("\nTraining completed!")

        return self.model
