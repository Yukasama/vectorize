"""Query and filter schemas for actions."""

from typing import Annotated

from fastapi import Query
from pydantic import BaseModel, Field

from vectorize.common.task_status import TaskStatus

__all__ = ["ActionQueryParams", "ActionsFilterParams"]


class ActionQueryParams(BaseModel):
    """Query parameters for actions endpoint."""

    limit: Annotated[int | None, Query(ge=1, le=100)] = None
    offset: Annotated[int | None, Query(ge=0)] = None
    completed: Annotated[
        bool | None, Query(description="Filter finished / unfinished")
    ] = None
    status: Annotated[
        list[TaskStatus] | None, Query(description="Filter by specific task statuses")
    ] = None
    within_hours: Annotated[
        int, Query(ge=1, description="Show tasks from last X hours")
    ] = 1


class ActionsFilterParams(BaseModel):
    """Parameters for filtering actions."""

    limit: int | None = Field(
        None, ge=1, le=100, description="Maximum number of records to return"
    )
    offset: int | None = Field(None, ge=0, description="Number of records to skip")
    completed: bool | None = Field(None, description="Filter by completion status")
    statuses: list[TaskStatus] | None = Field(
        None, description="Filter by specific task statuses"
    )
    within_hours: int = Field(
        1, ge=1, description="Time window in hours for task filtering"
    )
