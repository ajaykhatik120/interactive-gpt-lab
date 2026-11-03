import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim)
        )

    def forward(self, x, return_debug=False):
        if not return_debug:
            return self.network(x)

        linear_output = self.network[0](x)
        gelu_output = self.network[1](linear_output)
        final_output = self.network[2](gelu_output)

        debug = {
            "linear_output": linear_output,
            "gelu_output": gelu_output,
            "final_output": final_output
        }
        return final_output, debug
