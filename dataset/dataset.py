import torch
from torch.utils.data import Dataset


class GPTDataset(Dataset):

    def __init__(self, token_ids, block_size):

        self.token_ids = torch.tensor(
            token_ids,
            dtype=torch.long
        )

        self.block_size = block_size

        if len(self.token_ids) <= block_size:
            raise ValueError(
                f"Corpus has {len(self.token_ids)} tokens, "
                f"but block_size is {block_size}. "
                "Provide more training text or reduce block_size."
            )

    def __len__(self):

        return len(self.token_ids) - self.block_size

    def __getitem__(self, index):

        chunk = self.token_ids[
            index : index + self.block_size + 1
        ]

        x = chunk[:-1]
        y = chunk[1:]

        return x, y
