from dataclasses import dataclass


@dataclass
class GPTConfig:

    # Vocabulary and sequence
    vocab_size: int = 100
    block_size: int = 16

    # Transformer architecture
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.1

    # Training
    batch_size: int = 16
    learning_rate: float = 3e-4
    max_iters: int = 500
    eval_interval: int = 50


config = GPTConfig()