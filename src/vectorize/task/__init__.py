"""Task module for background task management and monitoring.

This module provides comprehensive task management capabilities for handling
asynchronous background processes across the Vectorize application, including
task creation, status tracking, filtering, and unified monitoring.

Key Components:
- Task aggregation: Unified view of tasks from multiple modules
  (model_upload, dataset_upload, training, evaluation, synthesis)
- Status management: Standardized task status tracking with real-time updates
- Filtering & querying: Advanced filtering by type, status, tag, and time windows
- Background processing: Async task execution with progress monitoring
- Database persistence: Task metadata and status storage with SQLModel

The task module serves as the central orchestrator for all background operations,
providing a consistent interface for task management across dataset uploads,
model training, evaluation processes, and data synthesis workflows.
"""

from .exceptions import TaskNotFoundError
from .models import TaskModel
from .query_builder import build_query
from .schemas import TaskFilters
from .task_status import TaskStatus
from .task_type import TaskType

__all__ = [
    "TaskFilters",
    "TaskModel",
    "TaskNotFoundError",
    "TaskStatus",
    "TaskType",
    "build_query",
]
