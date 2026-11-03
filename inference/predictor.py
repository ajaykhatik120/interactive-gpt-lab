
import torch


class GPTPredictor:

    def __init__(self, model, tokenizer):

        self.model = model
        self.tokenizer = tokenizer

        self.device = next(model.parameters()).device

    @torch.no_grad()
    def generate(
        self,
        prompt,
        max_new_tokens=20,
        temperature=1.0,
        top_k=None,
        greedy=False
    ):

        if temperature <= 0:
            raise ValueError("Temperature must be greater than zero.")

        # Convert prompt into token IDs
        token_ids = self.tokenizer.encode(prompt)

        if not token_ids:
            raise ValueError("Please provide a valid prompt.")

        # Add batch dimension
        idx = torch.tensor(
            [token_ids],
            dtype=torch.long,
            device=self.device
        )

        self.model.eval()

        for _ in range(max_new_tokens):

            # Keep only the most recent context tokens
            idx_cond = idx[:, -self.model.block_size:]

            # Forward pass
            logits, _ = self.model(idx_cond)

            # Get predictions from the final position
            logits = logits[:, -1, :]

            # Apply temperature
            logits = logits / temperature

            # Never generate the padding token
            pad_id = self.tokenizer.stoi.get("<PAD>")

            if pad_id is not None:
                logits[:, pad_id] = float("-inf")

            # Optional top-k filtering
            if top_k is not None:

                k = min(top_k, logits.size(-1))

                values, _ = torch.topk(logits, k)

                threshold = values[:, -1].unsqueeze(-1)

                logits = logits.masked_fill(
                    logits < threshold,
                    float("-inf")
                )

            # Convert logits into probabilities
            probabilities = torch.softmax(
                logits,
                dim=-1
            )

            # Select next token
            if greedy:

                next_token = torch.argmax(
                    probabilities,
                    dim=-1,
                    keepdim=True
                )

            else:

                next_token = torch.multinomial(
                    probabilities,
                    num_samples=1
                )

            # Append predicted token
            idx = torch.cat(
                (idx, next_token),
                dim=1
            )

            # Stop if EOS is generated
            eos_id = self.tokenizer.stoi.get("<EOS>")

            if eos_id is not None and next_token.item() == eos_id:
                break

        # Decode generated sequence
        return self.tokenizer.decode(idx[0].tolist())
