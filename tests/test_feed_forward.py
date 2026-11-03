import torch

from model.feed_forward import FeedForward
from config import config

# Simulate output from multi-head attention
x = torch.randn(1, 4, config.n_embd)

print("Input shape:", x.shape)

mlp = FeedForward(embedding_dim=config.n_embd)

output = mlp(x)

print("\nOutput shape:", output.shape)

print("\nFirst token representation:")
print(output[0, 0])

# Inspect the intermediate dimensions
expanded = mlp.network[0](x)

print("\nAfter first linear layer:")
print(expanded.shape)

activated = mlp.network[1](expanded)

print("\nAfter GELU:")
print(activated.shape)
