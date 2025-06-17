"""Orchestrates the end-to-end SBERT training process."""

from pathlib import Path
from uuid import UUID

from loguru import logger
from sentence_transformers import SentenceTransformer, losses
from sqlmodel.ext.asyncio.session import AsyncSession
from torch.utils.data import DataLoader

from vectorize.ai_model.model_source import ModelSource
from vectorize.ai_model.models import AIModel
from vectorize.ai_model.repository import get_ai_model_db, save_ai_model_db
from vectorize.common.task_status import TaskStatus
from vectorize.training.repository import update_training_task_validation_dataset

from .exceptions import DatasetValidationError
from .repository import get_train_task_by_id, update_training_task_status
from .schemas import TrainRequest
from .utils.cleanup import cleanup_resources
from .utils.input_examples import InputExampleDataset, prepare_input_examples
from .utils.model_loader import load_and_prepare_model
from .utils.validators import TrainingDataValidator


class TrainingOrchestrator:
    """Handles the orchestration of SBERT training."""

    def __init__(self, db: AsyncSession, task_id: UUID) -> None:
        """Initialize the training orchestrator.

        Args:
            db: Database session
            task_id: Training task ID
        """
        self.db = db
        self.task_id = task_id
        self.model: SentenceTransformer | None = None

    async def run_training(
        self,
        model_path: str,
        train_request: TrainRequest,
        dataset_paths: list[str],
        output_dir: str,
    ) -> None:
        """Main orchestration function for SBERT training.

        Args:
            model_path: Path to the base model
            train_request: Training configuration
            dataset_paths: List of dataset file paths
            output_dir: Output directory for the trained model
        """
        logger.debug(
            "Starting SBERT training",
            model_path=model_path,
            dataset_paths=dataset_paths,
            output_dir=output_dir,
            task_id=str(self.task_id),
        )

        try:
            self.model = load_and_prepare_model(model_path)

            (
                train_dataloader,
                validation_dataset_path,
            ) = TrainingOrchestrator._prepare_training_data(
                dataset_paths, train_request.per_device_train_batch_size
            )

            # Store validation dataset path in the training task
            await self._update_validation_dataset(validation_dataset_path)

            # Train the model and capture metrics
            training_metrics = self._train_model(train_dataloader, train_request, output_dir)

            # Save training metrics to database
            await self._save_training_metrics(training_metrics)

            await self._save_trained_model(train_request, output_dir)

            await update_training_task_status(self.db, self.task_id, TaskStatus.DONE)
            logger.debug(
                "Training finished successfully",
                model_path=model_path,
                task_id=str(self.task_id),
            )

        except (OSError, RuntimeError, DatasetValidationError) as exc:
            await self._handle_training_error(exc)
        finally:
            self._cleanup()

    @staticmethod
    def _prepare_training_data(
        dataset_paths: list[str], batch_size: int
    ) -> tuple[DataLoader, str | None]:
        """Prepare training data from multiple dataset paths.

        Args:
            dataset_paths: List of dataset file paths
                (training datasets + optional validation)
            batch_size: Training batch size

        Returns:
            Tuple of (DataLoader for training, validation dataset path)
        """
        if len(dataset_paths) > 1:
            return TrainingOrchestrator._prepare_multi_dataset_training(
                dataset_paths, batch_size
            )
        return TrainingOrchestrator._prepare_single_dataset_training(
            dataset_paths[0], batch_size
        )

    @staticmethod
    def _prepare_multi_dataset_training(
        dataset_paths: list[str], batch_size: int
    ) -> tuple[DataLoader, str | None]:
        """Prepare training data from multiple datasets with explicit validation."""
        training_paths = dataset_paths[:-1]
        validation_path = dataset_paths[-1]

        logger.info(
            "Multi-dataset training setup",
            num_training_datasets=len(training_paths),
            has_validation_dataset=True,
            training_datasets=training_paths,
            validation_dataset=validation_path,
        )

        # Combine all training datasets
        all_train_examples = []
        dataset_stats = []

        for i, path in enumerate(training_paths):
            df = TrainingDataValidator.validate_dataset(Path(path))
            examples = prepare_input_examples(df)
            all_train_examples.extend(examples)

            dataset_stats.append({
                "dataset_name": Path(path).name,
                "type": "training",
                "samples": len(df),
                "examples": len(examples),
            })

            logger.debug(
                "Loaded training dataset",
                dataset_index=i + 1,
                dataset_name=Path(path).name,
                samples=len(df),
                examples_generated=len(examples),
            )

        # Log validation dataset info
        val_df = TrainingDataValidator.validate_dataset(Path(validation_path))
        val_examples_count = len(prepare_input_examples(val_df))
        dataset_stats.append({
            "dataset_name": Path(validation_path).name,
            "type": "validation",
            "samples": len(val_df),
            "examples": val_examples_count,
        })

        TrainingOrchestrator._log_training_summary(
            dataset_stats, len(all_train_examples), batch_size, validation_path
        )

        train_dataset = InputExampleDataset(all_train_examples)
        return (
            DataLoader(train_dataset, batch_size=batch_size, num_workers=0),
            validation_path,
        )

    @staticmethod
    def _prepare_single_dataset_training(
        dataset_path: str, batch_size: int
    ) -> tuple[DataLoader, str | None]:
        """Prepare training data from single dataset with auto-split."""
        validation_dataset_path = f"{dataset_path}#auto-split"

        df = TrainingDataValidator.validate_dataset(Path(dataset_path))
        all_examples = prepare_input_examples(df)

        # 10% for validation, 90% for training
        val_split = int(0.1 * len(all_examples))
        train_examples = all_examples[val_split:]
        val_examples = all_examples[:val_split]

        dataset_stats = [{
            "dataset_name": Path(dataset_path).name,
            "type": "single_with_split",
            "total_samples": len(df),
            "total_examples": len(all_examples),
            "train_examples": len(train_examples),
            "validation_examples": len(val_examples),
            "validation_split": "10%",
        }]

        logger.info(
            "Single dataset training with auto-split",
            dataset_name=Path(dataset_path).name,
            total_samples=len(df),
            total_examples=len(all_examples),
            train_examples=len(train_examples),
            validation_examples=len(val_examples),
            validation_split_percent=10,
        )

        TrainingOrchestrator._log_training_summary(
            dataset_stats, len(train_examples), batch_size, validation_dataset_path
        )

        train_dataset = InputExampleDataset(train_examples)
        return (
            DataLoader(train_dataset, batch_size=batch_size, num_workers=0),
            validation_dataset_path,
        )

    @staticmethod
    def _log_training_summary(
        dataset_stats: list[dict],
        total_train_examples: int,
        batch_size: int,
        validation_dataset_path: str | None
    ) -> None:
        """Log final training data preparation summary."""
        # Convert dataset_stats to string to avoid Loguru formatting issues
        dataset_summary = []
        for stat in dataset_stats:
            summary = f"{stat.get('dataset_name', 'unknown')} ({stat.get('type', 'unknown')}): {stat.get('examples', 0)} examples"
            dataset_summary.append(summary)

        logger.info(
            "Training data preparation complete",
            total_datasets_used=len(dataset_stats),
            total_training_examples=total_train_examples,
            batch_size=batch_size,
            datasets=dataset_summary,
            validation_dataset_path=validation_dataset_path,
        )

    def _train_model(
        self,
        train_dataloader: DataLoader,
        train_request: TrainRequest,
        output_dir: str,
    ) -> dict:
        """Train the SBERT model.

        Args:
            train_dataloader: Training data loader
            train_request: Training configuration
            output_dir: Output directory for the trained model
            
        Returns:
            Dictionary containing training metrics
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        loss = losses.CosineSimilarityLoss(self.model)

        # Capture training start time for runtime calculation
        import builtins
        import time
        from typing import Any

        start_time = time.time()
        captured_metrics = {}

        # Monkey patch print to capture the metrics output
        original_print = builtins.print

        def custom_print(*args: Any, **kwargs: Any) -> None:
            """Custom print that captures training metrics."""
            # Convert args to string to check for metrics
            text = ' '.join(str(arg) for arg in args)

            # Check if this looks like the metrics dict
            if ('train_runtime' in text and 'train_loss' in text and
                'train_samples_per_second' in text):
                try:
                    # Try to parse as a Python dict
                    import ast
                    # Clean up the text - sometimes there might be extra content
                    if '{' in text and '}' in text:
                        start_idx = text.find('{')
                        end_idx = text.rfind('}') + 1
                        dict_str = text[start_idx:end_idx]
                        parsed_metrics = ast.literal_eval(dict_str)
                        if isinstance(parsed_metrics, dict):
                            captured_metrics.update(parsed_metrics)
                            logger.info("Captured training metrics from print", **parsed_metrics)
                except (ValueError, SyntaxError) as e:
                    logger.debug("Failed to parse metrics from print", text=text, error=str(e))

            # Call original print
            original_print(*args, **kwargs)

        # Replace print temporarily
        builtins.print = custom_print

        try:
            # Train the model
            self.model.fit(
                train_objectives=[(train_dataloader, loss)],
                epochs=train_request.epochs,
                warmup_steps=train_request.warmup_steps or 0,
                show_progress_bar=False,
                output_path=str(Path(output_dir)),
            )
        finally:
            # Restore original print
            builtins.print = original_print

        # Calculate training runtime
        end_time = time.time()
        train_runtime = end_time - start_time

        # Use captured metrics if available, otherwise fall back to calculated
        try:
            total_samples = len(train_dataloader.dataset)  # type: ignore
        except (TypeError, AttributeError):
            total_samples = len(train_dataloader) * train_request.per_device_train_batch_size

        total_steps = len(train_dataloader) * train_request.epochs

        training_metrics = {
            'train_runtime': captured_metrics.get('train_runtime', train_runtime),
            'train_samples_per_second': captured_metrics.get('train_samples_per_second',
                                                           total_samples / train_runtime if train_runtime > 0 else 0.0),
            'train_steps_per_second': captured_metrics.get('train_steps_per_second',
                                                         total_steps / train_runtime if train_runtime > 0 else 0.0),
            'train_loss': captured_metrics.get('train_loss', 0.0),
            'epoch': captured_metrics.get('epoch', float(train_request.epochs))
        }

        # Log what we captured
        if captured_metrics:
            logger.info("Using captured training metrics", **captured_metrics)
        else:
            logger.debug("No metrics captured, using calculated values", calculated_runtime=train_runtime)

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.model.save(str(output_dir))

        return training_metrics

    async def _save_trained_model(
        self, train_request: TrainRequest, output_dir: str
    ) -> None:
        """Save the trained model to the database.

        Args:
            train_request: Training configuration
            output_dir: Output directory where model was saved
        """
        parent_model = await get_ai_model_db(self.db, train_request.model_tag)
        tag_time = Path(output_dir).name
        new_model_tag = str(Path(output_dir).relative_to("data/models"))

        new_model = AIModel(
            name=f"Fine-tuned: {parent_model.name} {tag_time}",
            model_tag=new_model_tag,
            source=ModelSource.LOCAL,
            trained_from_id=parent_model.id,
            trained_from_tag=parent_model.model_tag,
        )

        new_model_id = await save_ai_model_db(self.db, new_model)

        task = await get_train_task_by_id(self.db, self.task_id)
        if task:
            task.trained_model_id = new_model_id
            await self.db.commit()
            await self.db.refresh(task)

    async def _save_training_metrics(self, training_metrics: dict) -> None:
        """Save training metrics to the database.
        
        Args:
            training_metrics: Dictionary containing training metrics
        """
        from .repository import update_training_task_metrics

        await update_training_task_metrics(
            self.db,
            self.task_id,
            train_runtime=training_metrics.get('train_runtime'),
            train_samples_per_second=training_metrics.get('train_samples_per_second'),
            train_steps_per_second=training_metrics.get('train_steps_per_second'),
            train_loss=training_metrics.get('train_loss'),
            epoch=training_metrics.get('epoch'),
        )

        logger.debug(
            "Training metrics saved to database",
            task_id=str(self.task_id),
            **training_metrics
        )

    async def _handle_training_error(self, exc: Exception) -> None:
        """Handle training errors.

        Args:
            exc: The exception that occurred
        """
        logger.error(
            "Training failed",
            task_id=str(self.task_id),
            exc=str(exc),
        )
        await update_training_task_status(
            self.db,
            self.task_id,
            TaskStatus.FAILED,
            error_msg=str(exc),
        )

    async def _update_validation_dataset(
        self, validation_dataset_path: str | None
    ) -> None:
        """Update the training task with the validation dataset path.

        Args:
            validation_dataset_path: Path to the validation dataset used during training
        """
        if validation_dataset_path:
            await update_training_task_validation_dataset(
                self.db, self.task_id, validation_dataset_path
            )

            logger.debug(
                "Updated training task with validation dataset",
                task_id=str(self.task_id),
                validation_dataset_path=validation_dataset_path,
            )

    def _cleanup(self) -> None:
        """Clean up resources."""
        if self.model is not None:
            cleanup_resources(self.model)
            self.model = None


async def run_training(  # noqa: PLR0913, PLR0917
    db: AsyncSession,
    model_path: str,
    train_request: TrainRequest,
    task_id: UUID,
    dataset_paths: list[str],
    output_dir: str,
) -> None:
    """Legacy wrapper function for backward compatibility.

    Args:
        db: Database session
        model_path: Path to the base model
        train_request: Training configuration
        task_id: Training task ID
        dataset_paths: List of dataset file paths
        output_dir: Output directory for the trained model
    """
    orchestrator = TrainingOrchestrator(db, task_id)
    await orchestrator.run_training(
        model_path, train_request, dataset_paths, output_dir
    )
