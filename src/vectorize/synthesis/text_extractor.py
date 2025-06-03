"""Text extraction from media files (images and PDFs)."""

from pathlib import Path

import pandas as pd
from loguru import logger

from vectorize.datasets.upload_options_model import DatasetUploadOptions

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

    question_col = (
        options.question_name if options and options.question_name else "question"
    )
    positive_col = (
        options.positive_name if options and options.positive_name else "positive"
    )
    negative_col = (
        options.negative_name if options and options.negative_name else "negative"
    )

    if file_type.lower() == "pdf":
        data = _extract_from_pdf(file_path, question_col, positive_col, negative_col)
    elif file_type.lower() == "dataset":
        data = _extract_from_dataset(
            file_path, question_col, positive_col, negative_col
        )
    else:
        data = _extract_from_image(file_path, question_col, positive_col, negative_col)

    logger.debug(f"Extracted {len(data)} text elements from {file_type} file")
    return pd.DataFrame(data)


def _extract_from_pdf(
    file_path: Path, question_col: str, positive_col: str, negative_col: str
) -> list[dict[str, str]]:
    """Extract text from PDF file (placeholder implementation).

    Args:
        file_path: Path to the PDF file
        question_col: Name for question column
        positive_col: Name for positive column
        negative_col: Name for negative column

    Returns:
        List of dictionaries containing extracted text data
    """
    logger.debug(f"Processing PDF file: {file_path}")

    return [
        {
            question_col: f"What is discussed on page {i}?",
            positive_col: f"Content extracted from page {i} of the PDF document.",
            negative_col: f"Unrelated information about page {i}.",
        }
        for i in range(1, 4)
    ]


def _extract_from_image(
    file_path: Path, question_col: str, positive_col: str, negative_col: str
) -> list[dict[str, str]]:
    """Extract text from image file (placeholder implementation).

    Args:
        file_path: Path to the image file
        question_col: Name for question column
        positive_col: Name for positive column
        negative_col: Name for negative column

    Returns:
        List of dictionaries containing extracted text data
    """
    logger.debug(f"Processing image file: {file_path}")

    return [
        {
            question_col: "What does this image show?",
            positive_col: "Text extracted from image using OCR technology.",
            negative_col: "Random sentence unrelated to the image content.",
        },
        {
            question_col: "Is there any text visible in this image?",
            positive_col: "Yes, the extracted text is displayed in this dataset.",
            negative_col: "This is an unrelated sentence about something else.",
        },
        {
            question_col: f"What can you tell me about {file_path.stem}?",
            positive_col: (
                f"This image file named {file_path.stem} contains readable text."
            ),
            negative_col: "Random information not related to the image analysis.",
        },
    ]


def _extract_from_dataset(
    file_path: Path, question_col: str, positive_col: str, negative_col: str
) -> list[dict[str, str]]:
    """Extract synthetic data based on existing dataset content.

    Args:
        file_path: Path to the dataset CSV file
        question_col: Name for question column
        positive_col: Name for positive column
        negative_col: Name for negative column

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
            question_col: "What patterns can be found in the source dataset?",
            positive_col: (
                f"The dataset contains {num_rows} examples "
                f"with structured question-answer pairs."
            ),
            negative_col: (
                "Random unrelated content that doesn't match the dataset structure."
            ),
        },
        {
            question_col: "How can we expand this dataset synthetically?",
            positive_col: (
                "By analyzing existing patterns and generating similar "
                "structured examples."
            ),
            negative_col: (
                "By copying random text from unrelated sources without any structure."
            ),
        },
        {
            question_col: "What makes this synthetic extension valuable?",
            positive_col: (
                "It maintains the original dataset's format "
                "while adding new training examples."
            ),
            negative_col: (
                "Synthetic data is always inferior and should never be used "
                "for training."
            ),
        },
        {
            question_col: "Why use the source dataset as a foundation?",
            positive_col: (
                f"It ensures consistency with the original {num_rows} examples "
                f"and their structure."
            ),
            negative_col: (
                "Starting from scratch would be more efficient "
                "than using existing data."
            ),
        },
    ]
