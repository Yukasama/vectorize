"""Background task for model training."""

from datetime import UTC, datetime
from uuid import UUID

from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.ai_model.model_source import ModelSource
from vectorize.ai_model.models import AIModel
from vectorize.ai_model.repository import get_ai_model_by_id, save_ai_model_db
from vectorize.common.task_status import TaskStatus

from .repository import get_train_task_by_id, update_training_task_status
from .schemas import TrainRequest
from .service import train_model_service_svc
from .utils.uuid_utils import is_valid_uuid, normalize_uuid
from .exceptions import InvalidModelIdError


async def train_model_task(
    db: AsyncSession,
    model_path: str,
    train_request: TrainRequest,
    task_id: UUID,
    dataset_paths: list[str],
) -> None:
    """Background task: trains the model, saves new AIModel, updates TrainingTask."""
    logger.info(
        "Training started for model_path={}, dataset_paths={}, task_id={}",
        model_path,
        dataset_paths,
        task_id,
    )
    try:
        # Validierung und Normalisierung der Modell-ID (defensiv)
        if not is_valid_uuid(train_request.model_id):
            raise InvalidModelIdError(train_request.model_id)
        norm_model_id = UUID(train_request.model_id)
        orig_model = await get_ai_model_by_id(db, norm_model_id)
        train_model_service_svc(model_path, train_request, dataset_paths)

        tag_time = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        new_model_tag = f"{orig_model.model_tag}-finetuned-{tag_time}"
        new_model = AIModel(
            name=f"Fine-tuned: {orig_model.name} {tag_time}",
            model_tag=new_model_tag,
            source=ModelSource.LOCAL,
            trained_from_id=orig_model.id,
        )
        new_model_id = await save_ai_model_db(db, new_model)
        task = await get_train_task_by_id(db, task_id)
        if task:
            task.trained_model_id = new_model_id
            await db.commit()
            await db.refresh(task)
        await update_training_task_status(db, task_id, TaskStatus.DONE)
        logger.info(
            "Training finished successfully for model_path={}, task_id={}, "
            "new_model_id={}",
            model_path,
            task_id,
            new_model_id,
        )
    except Exception as exc:
        logger.exception(f"Training failed: task_id={task_id} - {exc}")
        await update_training_task_status(
            db, task_id, TaskStatus.FAILED, error_msg=str(exc)
        )
