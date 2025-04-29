"""Mean pooling for transformer embeddings."""

import torch

__all__ = ["mean_pool"]


def mean_pool(
    last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Perform mean pooling on transformer embeddings.

    Args:
        last_hidden_state: Hidden state tensor from transformer model.
            Shape is typically (batch_size, sequence_length, hidden_size).
        attention_mask: Attention mask tensor indicating valid tokens vs padding.
            Shape is typically (batch_size, sequence_length) with 1s for tokens
            and 0s for padding.

    Returns:
        Mean-pooled embedding tensor with shape (batch_size, hidden_size).
    """
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * mask).sum(1) / mask.sum(1)
