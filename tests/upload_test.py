from unittest.mock import MagicMock, patch

import pytest

from txt2vec.upload.model_service import get_classifier, load_model_with_tag


@pytest.fixture
def mock_snapshot_download():
    with patch("txt2vec.upload.model_service.snapshot_download") as mock:
        mock.return_value = "/mocked/path/to/model"
        yield mock


@pytest.fixture
def mock_auto_tokenizer():
    with patch("txt2vec.upload.model_service.AutoTokenizer.from_pretrained") as mock:
        mock.return_value = MagicMock()
        yield mock


@pytest.fixture
def mock_auto_model():
    with patch(
        "txt2vec.upload.model_service.AutoModelForSequenceClassification.from_pretrained"
    ) as mock:
        mock.return_value = MagicMock()
        yield mock


@pytest.fixture
def mock_pipeline():
    with patch("txt2vec.upload.model_service.pipeline") as mock:
        mock.return_value = MagicMock()
        yield mock


def test_load_model_with_tag(
    mock_snapshot_download, mock_auto_tokenizer, mock_auto_model, mock_pipeline
):
    # Arrange
    model_id = "distilbert-base-uncased-finetuned-sst-2-english"
    tag = "main"

    # Act
    load_model_with_tag(model_id, tag)

    # Assert
    mock_snapshot_download.assert_called_once_with(
        repo_id=model_id, revision=tag, cache_dir="./hf_cache"
    )
    mock_auto_tokenizer.assert_called_once_with("/mocked/path/to/model")
    mock_auto_model.assert_called_once_with("/mocked/path/to/model")
    mock_pipeline.assert_called_once_with(
        "sentiment-analysis",
        model=mock_auto_model.return_value,
        tokenizer=mock_auto_tokenizer.return_value,
    )


def test_get_classifier_without_loading():
    # Arrange
    from txt2vec.upload.model_service import reset_classifier

    reset_classifier()  # Setze CLASSIFIER explizit auf None

    # Act & Assert
    with pytest.raises(ValueError, match="Kein Modell geladen."):
        get_classifier()


def test_get_classifier_after_loading(
    mock_snapshot_download, mock_auto_tokenizer, mock_auto_model, mock_pipeline
):
    # Arrange
    model_id = "distilbert-base-uncased-finetuned-sst-2-english"
    tag = "main"
    load_model_with_tag(model_id, tag)

    # Act
    classifier = get_classifier()

    # Assert
    assert classifier == mock_pipeline.return_value
