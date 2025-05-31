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
    load_and_tokenize_datasets,
    load_model_and_tokenizer,
    prepare_output_dir,
    train,
)


def train_model_service_svc(model_id: str, train_request: TrainRequest) -> None:
    """Trains a model with triplet loss on local CSV datasets.

    Loads model and tokenizer, processes multiple datasets, performs training
    with TripletMarginLoss, and saves the model.
    """
    with logger.contextualize(model_id=model_id):
        logger.info("Training started.")
        if missing := [p for p in train_request.dataset_paths if not Path(p).is_file()]:
            logger.error("Training failed: Dataset file(s) not found: %s", missing)
            raise TrainingDatasetNotFoundError(", ".join(missing))
        output_dir = prepare_output_dir(model_id)
        model, tokenizer = load_model_and_tokenizer(model_id)
        dataloader = load_and_tokenize_datasets(
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
        train(
            train_ctx,
            train_request.epochs,
            output_dir=output_dir,
            checkpoint_interval=1,
        )
        with torch.no_grad():
            model.save_pretrained(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))
