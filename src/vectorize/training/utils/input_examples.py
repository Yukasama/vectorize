"""Utility for converting DataFrames to InputExamples for SBERT training."""

import pandas as pd
from sentence_transformers import InputExample
from torch.utils.data import Dataset


class InputExampleDataset(Dataset):
    """Wraps a list of InputExamples for use with DataLoader.

    Args:
        examples (list): List of InputExample objects.
    """

    def __init__(self, examples: list) -> None:
        """Initializes the dataset.

        Args:
            examples (list): List of InputExample objects.
        """
        self.examples = examples

    def __getitem__(self, idx: int) -> InputExample:
        """Returns the InputExample at the given index.

        Args:
            idx (int): Index of the example.

        Returns:
            InputExample: The example at the given index.
        """
        return self.examples[idx]

    def __len__(self) -> int:
        """Returns the number of examples in the dataset.

        Returns:
            int: Number of examples.
        """
        return len(self.examples)


def prepare_input_examples(df: pd.DataFrame) -> list[InputExample]:
    """Convert a DataFrame with 'Question', 'Positive', 'Negative' columns.

    Args:
        df (pd.DataFrame): DataFrame with required columns.

    Returns:
        list[InputExample]: List of InputExample objects for SBERT training.
    """
    examples = []
    for _, row in df.iterrows():
        q = str(row["Question"])
        pos = str(row["Positive"])
        neg = str(row["Negative"])
        examples.extend([
            InputExample(texts=[q, pos], label=1.0),
            InputExample(texts=[q, neg], label=-1.0),
        ])
    return examples
