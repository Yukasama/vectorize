# ruff: noqa: S101, PLR6301

"""Tests for evaluation service integration with training tasks."""

from pathlib import Path
from uuid import uuid4

import pytest
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.evaluation.schemas import EvaluationRequest
from vectorize.evaluation.service import resolve_evaluation_dataset
from vectorize.training.models import TrainingTask
from vectorize.training.repository import (
    save_training_task,
    update_training_task_validation_dataset,
)


class TestEvaluationIntegration:
    """Test integration between evaluation and training systems."""

    @pytest.mark.asyncio
    async def test_resolve_dataset_with_training_task_id(
        self, session: AsyncSession
    ) -> None:
        """Test resolving dataset using training_task_id."""
        # Create test training task
        task = TrainingTask(
            id=uuid4(),
            model_tag="test-model"
        )
        await save_training_task(session, task)

        # Create validation dataset file
        validation_path = Path("data/datasets/validation_test.jsonl")
        validation_path.parent.mkdir(parents=True, exist_ok=True)
        validation_path.write_text(
            '{"question": "test", "positive": "pos", "negative": "neg"}\n'
        )

        # Update task with validation dataset path
        await update_training_task_validation_dataset(
            session, task.id, str(validation_path)
        )

        try:
            # Test request with training_task_id
            request = EvaluationRequest(
                model_tag="test-model",
                training_task_id=str(task.id),
                max_samples=100
            )

            result_path = await resolve_evaluation_dataset(session, request)
            assert result_path == validation_path

        finally:
            # Cleanup
            if validation_path.exists():
                validation_path.unlink()

    @pytest.mark.asyncio
    async def test_resolve_dataset_with_both_ids_fails(
        self, session: AsyncSession
    ) -> None:
        """Test that providing both dataset_id and training_task_id fails."""
        request = EvaluationRequest(
            model_tag="test-model",
            dataset_id=str(uuid4()),
            training_task_id=str(uuid4()),
            max_samples=100
        )

        with pytest.raises(ValueError, match="Cannot specify both"):
            await resolve_evaluation_dataset(session, request)

    @pytest.mark.asyncio
    async def test_resolve_dataset_with_no_ids_fails(
        self, session: AsyncSession
    ) -> None:
        """Test that providing neither dataset_id nor training_task_id fails."""
        request = EvaluationRequest(
            model_tag="test-model",
            max_samples=100
        )

        with pytest.raises(ValueError, match="Must specify either"):
            await resolve_evaluation_dataset(session, request)
