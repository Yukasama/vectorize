"""Utility for converting DataFrames to InputExamples for SBERT training."""

import pandas as pd
from sentence_transformers import InputExample


def prepare_input_examples(df: pd.DataFrame) -> list[InputExample]:
    """Convert a DataFrame with 'Question', 'Positive', 'Negative' columns to InputExamples.
    """
    examples = []
    for _, row in df.iterrows():
        q, pos, neg = row["Question"], row["Positive"], row["Negative"]
        examples.extend([
            InputExample(texts=[q, pos], label=1.0),
            InputExample(texts=[q, neg], label=-1.0),
        ])
    return examples
