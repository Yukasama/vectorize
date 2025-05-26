"""Service for model training (Transformers Trainer API)."""

from pathlib import Path

import torch
from loguru import logger
from torch.nn import TripletMarginLoss

from .exceptions import (
    TrainingDatasetNotFoundError,
)
from .schemas import TrainRequest
from .utils.helpers import (
    _load_and_tokenize_datasets,
    _load_model_and_tokenizer,
    _prepare_output_dir,
    _train,
)


def train_model_service_svc(train_request: TrainRequest) -> None:
    """Trainiert ein Modell mit Triplet-Loss auf lokalen CSV-Datensätzen.

    Lädt Modell und Tokenizer, verarbeitet mehrere Datasets, führt das Training
    mit TripletMarginLoss durch und speichert das Modell.
    """
    with logger.contextualize(model_path=train_request.model_path):
        logger.info("Training started.")
        if missing := [p for p in train_request.dataset_paths if not Path(p).is_file()]:
            logger.error("Training failed: Dataset file(s) not found: %s", missing)
            raise TrainingDatasetNotFoundError(", ".join(missing))
        output_dir = _prepare_output_dir(train_request.model_path)
        model, tokenizer = _load_model_and_tokenizer(train_request.model_path)
        dataloader = _load_and_tokenize_datasets(
            train_request.dataset_paths,
            tokenizer,
            train_request.per_device_train_batch_size,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_request.learning_rate,
        )
        criterion = TripletMarginLoss(margin=1.0, p=2)
        train_ctx = {
            "model": model,
            "dataloader": dataloader,
            "optimizer": optimizer,
            "criterion": criterion,
            "device": device,
        }
        _train(train_ctx, train_request.epochs)
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
