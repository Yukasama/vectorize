"""Dataset utilities for triplet training."""

import torch
from torch.utils.data import Dataset


class TripletDataset(Dataset):
    """PyTorch Dataset for triplet (anchor, positive, negative) training."""
    def __init__(self, data: dict) -> None:
        """Initialisiert das TripletDataset mit den gegebenen Daten."""
        self.data = data

    def __len__(self) -> int:
        """Gibt die Anzahl der Triplets im Dataset zurück."""
        return len(self.data["anchor_input_ids"])

    def __getitem__(self, idx: int) -> dict:
        """Gibt das Triplet an Index idx als Dictionary mit Tensors zurück."""
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
        """String-Repräsentation des TripletDataset."""
        return f"<TripletDataset samples={len(self)}>"


def preprocess_triplet_batch(tokenizer: object, examples: dict) -> dict:
    """Tokenisiert ein Batch von Triplet-Beispielen für das Training.

    Raises:
        ValueError: Wenn ein erforderlicher Key fehlt.
    """
    try:
        anchor = tokenizer(examples["question"], truncation=True, padding=True)
        positive = tokenizer(examples["positive"], truncation=True, padding=True)
        negative = tokenizer(examples["negative"], truncation=True, padding=True)
    except KeyError as e:
        raise ValueError(f"Missing key in input batch: {e}") from e
    return {
        "anchor_input_ids": anchor["input_ids"],
        "anchor_attention_mask": anchor["attention_mask"],
        "positive_input_ids": positive["input_ids"],
        "positive_attention_mask": positive["attention_mask"],
        "negative_input_ids": negative["input_ids"],
        "negative_attention_mask": negative["attention_mask"],
    }
