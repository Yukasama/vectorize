"""Text extraction from media files (images and PDFs)."""

from pathlib import Path

import pandas as pd
from loguru import logger

from vectorize.dataset.upload_options_model import DatasetUploadOptions

__all__ = ["extract_text_from_media"]


def extract_text_from_media(
    file_path: Path, file_type: str, options: DatasetUploadOptions | None = None
) -> pd.DataFrame:
    """Extract text from images or PDF files and convert to a structured DataFrame.

    This is a placeholder function that creates synthetic data based on file type.
    In a real implementation, this would use OCR for images and PDF parsing for PDFs.

    Args:
        file_path: Path to the media file.
        file_type: Type of the file (pdf, png, jpg, jpeg, dataset).
        options: Optional dataset upload options for column naming.

    Returns:
        pd.DataFrame: DataFrame with extracted text in a structure compatible
                     with the dataset system.

    Raises:
        ValueError: If file_type is not supported.
    """
    logger.info(f"Extracting text from {file_type} file: {file_path}")

    # Validate file type
    supported_types = {"pdf", "png", "jpg", "jpeg", "dataset"}
    if file_type.lower() not in supported_types:
        raise ValueError(
            f"Unsupported file type: {file_type}. Supported: {supported_types}"
        )

    prompt_col = options.prompt_name if options and options.prompt_name else "prompt"
    chosen_col = options.chosen_name if options and options.chosen_name else "chosen"
    rejected_col = (
        options.rejected_name if options and options.rejected_name else "rejected"
    )

    if file_type.lower() == "pdf":
        data = _extract_from_pdf(file_path, prompt_col, chosen_col, rejected_col)
    elif file_type.lower() == "dataset":
        data = _extract_from_dataset(file_path, prompt_col, chosen_col, rejected_col)
    else:
        data = _extract_from_image(file_path, prompt_col, chosen_col, rejected_col)

    logger.debug(f"Extracted {len(data)} text elements from {file_type} file")
    return pd.DataFrame(data)


def _extract_from_pdf(
    file_path: Path, prompt_col: str, chosen_col: str, rejected_col: str
) -> list[dict[str, str]]:
    """Extract text from PDF file (placeholder implementation).

    Args:
        file_path: Path to the PDF file
        prompt_col: Name for prompt column
        chosen_col: Name for chosen column
        rejected_col: Name for rejected column

    Returns:
        List of dictionaries containing extracted text data
    """
    logger.debug(f"Processing PDF file: {file_path}")

    return [
        {
            prompt_col: f"What is discussed on page {i}?",
            chosen_col: f"Content extracted from page {i} of the PDF document.",
            rejected_col: f"Unrelated information about page {i}.",
        }
        for i in range(1, 4)
    ]


def _extract_from_image(
    file_path: Path, prompt_col: str, chosen_col: str, rejected_col: str
) -> list[dict[str, str]]:
    """Extract text from image file (placeholder implementation).

    Args:
        file_path: Path to the image file
        prompt_col: Name for prompt column
        chosen_col: Name for chosen column
        rejected_col: Name for rejected column

    Returns:
        List of dictionaries containing extracted text data
    """
    logger.debug(f"Processing image file: {file_path}")

    return [
        {
            prompt_col: "What does this image show?",
            chosen_col: "Text extracted from image using OCR technology.",
            rejected_col: "Random sentence unrelated to the image content.",
        },
        {
            prompt_col: "Is there any text visible in this image?",
            chosen_col: "Yes, the extracted text is displayed in this dataset.",
            rejected_col: "This is an unrelated sentence about something else.",
        },
        {
            prompt_col: f"What can you tell me about {file_path.stem}?",
            chosen_col: (
                f"This image file named {file_path.stem} contains readable text."
            ),
            rejected_col: "Random information not related to the image analysis.",
        },
    ]


def _extract_from_dataset(
    file_path: Path, prompt_col: str, chosen_col: str, rejected_col: str
) -> list[dict[str, str]]:
    """Extract synthetic data based on existing dataset content.

    Args:
        file_path: Path to the dataset CSV file
        prompt_col: Name for prompt column
        chosen_col: Name for chosen column
        rejected_col: Name for rejected column

    Returns:
        List of dictionaries containing synthetic dataset data
    """
    logger.debug(f"Processing dataset file: {file_path}")

    try:
        existing_df = pd.read_csv(file_path)

        logger.debug(
            "Loaded existing dataset for synthesis",
            fileName=file_path.name,
            rows=len(existing_df),
            columns=list(existing_df.columns),
        )

        num_rows = len(existing_df)

    except Exception as e:
        logger.warning(f"Could not load dataset file {file_path}: {e}")
        num_rows = 3  # Fallback

    return [
        {
            prompt_col: "What patterns can be found in the source dataset?",
            chosen_col: (
                f"The dataset contains {num_rows} examples "
                f"with structured prompt-answer pairs."
            ),
            rejected_col: (
                "Random unrelated content that doesn't match the dataset structure."
            ),
        },
        {
            prompt_col: "How can we expand this dataset synthetically?",
            chosen_col: (
                "By analyzing existing patterns and generating similar "
                "structured examples."
            ),
            rejected_col: (
                "By copying random text from unrelated sources without any structure."
            ),
        },
        {
            prompt_col: "What makes this synthetic extension valuable?",
            chosen_col: (
                "It maintains the original dataset's format "
                "while adding new training examples."
            ),
            rejected_col: (
                "Synthetic data is always inferior and should never be used "
                "for training."
            ),
        },
        {
            prompt_col: "Why use the source dataset as a foundation?",
            chosen_col: (
                f"It ensures consistency with the original {num_rows} examples "
                f"and their structure."
            ),
            rejected_col: (
                "Starting from scratch would be more efficient "
                "than using existing data."
            ),
        },
    ]
