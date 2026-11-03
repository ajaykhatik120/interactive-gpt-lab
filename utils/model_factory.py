from dataset.dataset import GPTDataset
from model.gpt import GPTLanguageModel
from config import config


def build_model_and_dataset(tokenizer, token_ids):
    vocab_size = tokenizer.vocab_size
    total_tokens = len(token_ids)

    if total_tokens < 2:
        raise ValueError(
            "The story must produce at least two encoded tokens."
        )

    block_size = min(config.block_size, total_tokens - 1)

    dataset = GPTDataset(
        token_ids=token_ids,
        block_size=block_size
    )

    model = GPTLanguageModel(
        vocab_size=vocab_size,
        embedding_dim=config.n_embd,
        num_heads=config.n_head,
        num_layers=config.n_layer,
        block_size=block_size
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return {
        "model": model,
        "dataset": dataset,
        "vocab_size": vocab_size,
        "block_size": block_size,
        "dataset_size": len(dataset),
        "trainable_parameters": trainable_parameters
    }
