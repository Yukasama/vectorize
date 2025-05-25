"""Test."""
# ruff: noqa: S101
# ruff noqa: PLR2004

import math
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlmodel.ext.asyncio.session import AsyncSession

from vectorize.ai_model import AIModel, ModelSource

DEFAULT_PAGE_SIZE: int = 10
DEFAULT_PAGE_NUMBER: int = 1


@pytest.mark.asyncio
async def test_list_models_pagination_non_default_params(
    session: AsyncSession,
    client: TestClient,
) -> None:
    """Test models paged."""
    dbobjectcount: int = 23

    models = [
        AIModel(
            id=uuid4(),
            version=0,
            name=f"Model_{i}",
            model_tag=f"tag_{i}_For_Test",
            source=ModelSource.LOCAL,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        for i in range(dbobjectcount)
    ]
    session.add_all(models)
    await session.commit()

    last_pagenum = math.ceil(dbobjectcount / DEFAULT_PAGE_SIZE)
    expected_last_page_items = dbobjectcount % DEFAULT_PAGE_SIZE or DEFAULT_PAGE_SIZE

    # First page
    route = f"/models?page={DEFAULT_PAGE_NUMBER}&size={DEFAULT_PAGE_SIZE}"
    resp = client.get(route)
    assert resp.status_code == status.HTTP_200_OK
    payload = resp.json()
    assert payload["page"] == DEFAULT_PAGE_NUMBER
    assert payload["size"] == DEFAULT_PAGE_SIZE
    assert payload["totalpages"] == last_pagenum
    assert len(payload["items"]) == DEFAULT_PAGE_SIZE

    # Last page
    route = f"/models?page={last_pagenum}&size={DEFAULT_PAGE_SIZE}"
    resp = client.get(route)
    assert resp.status_code == status.HTTP_200_OK
    payload = resp.json()
    assert payload["page"] == last_pagenum
    assert len(payload["items"]) == expected_last_page_items

    names = [item["name"] for item in payload["items"]]
    assert names[0].startswith("Model_")


@pytest.mark.asyncio
async def test_list_models_pagination_default_params(
    session: AsyncSession,
    client: TestClient,
) -> None:
    """Test models paged."""
    dbobjectcount: int = 23
    last_pagenum = math.ceil(dbobjectcount / DEFAULT_PAGE_SIZE)
    expected_last_page_items = dbobjectcount % DEFAULT_PAGE_SIZE or DEFAULT_PAGE_SIZE

    # Die Seeddaten müssen wie im oberen Test vorbereitet werden!
    models = [
        AIModel(
            id=uuid4(),
            version=0,
            name=f"Model_{i}",
            model_tag=f"tag_{i}_For_Test_DEFAULT",
            source=ModelSource.LOCAL,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        for i in range(dbobjectcount)
    ]
    session.add_all(models)
    await session.commit()

    # First page
    route = f"/models?page={DEFAULT_PAGE_NUMBER}&size={DEFAULT_PAGE_SIZE}"
    resp = client.get(route)
    assert resp.status_code == status.HTTP_200_OK
    payload = resp.json()
    assert payload["page"] == DEFAULT_PAGE_NUMBER
    assert payload["size"] == DEFAULT_PAGE_SIZE
    assert payload["totalpages"] == last_pagenum
    assert len(payload["items"]) == DEFAULT_PAGE_SIZE

    # Last page
    route = f"/models?page={last_pagenum}&size={DEFAULT_PAGE_SIZE}"
    resp = client.get(route)
    assert resp.status_code == status.HTTP_200_OK
    payload = resp.json()
    assert payload["page"] == last_pagenum
    assert len(payload["items"]) == expected_last_page_items

    names = [item["name"] for item in payload["items"]]
    assert names[0].startswith("Model_")
