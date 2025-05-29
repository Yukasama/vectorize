"""Dataset utilities for triplet training."""

import torch
from torch.utils.data import Dataset


class TripletDataset(Dataset):
    """PyTorch Dataset for triplet (anchor, positive, negative) training."""
    def __init__(self, data: dict) -> None:
        """Initializes the TripletDataset with the given data."""
        self.data = data

    def __len__(self) -> int:
        """Returns the number of triplets in the dataset."""
        return len(self.data["anchor_input_ids"])

    def __getitem__(self, idx: int) -> dict:
        """Returns the triplet at index idx as a dictionary of tensors."""
        return {
            "anchor_input_ids": torch.tensor(
                self.data["anchor_input_ids"][idx]
            ),
            "anchor_attention_mask": torch.tensor(
                self.data["anchor_attention_mask"][idx]
            ),
            "positive_input_ids": torch.tensor(
                self.data["positive_input_ids"][idx]
            ),
            "positive_attention_mask": torch.tensor(
                self.data["positive_attention_mask"][idx]
            ),
            "negative_input_ids": torch.tensor(
                self.data["negative_input_ids"][idx]
            ),
            "negative_attention_mask": torch.tensor(
                self.data["negative_attention_mask"][idx]
            ),
        }

    def __repr__(self) -> str:
        """String representation of the TripletDataset."""
        return f"<TripletDataset samples={len(self)}>"


def preprocess_triplet_batch(tokenizer: object, examples: dict) -> dict:
    """Tokenizes a batch of triplet examples for training.

    Raises:
        ValueError: If a required key is missing.
    """
    try:
        anchor = tokenizer(
            examples["question"],
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            max_length=128,
        )
        positive = tokenizer(
            examples["positive"],
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            max_length=128,
        )
        negative = tokenizer(
            examples["negative"],
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            max_length=128,
        )
    except KeyError as e:
        raise ValueError(f"Missing key in input batch: {e}") from e
    return {
        "anchor_input_ids": anchor["input_ids"].tolist(),
        "anchor_attention_mask": anchor["attention_mask"].tolist(),
        "positive_input_ids": positive["input_ids"].tolist(),
        "positive_attention_mask": positive["attention_mask"].tolist(),
        "negative_input_ids": negative["input_ids"].tolist(),
        "negative_attention_mask": negative["attention_mask"].tolist(),
    }
