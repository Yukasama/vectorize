"""Test to check the default and non default returns of the paged models endpoint."""
# ruff: noqa: S101
# ruff noqa: PLR2004

import math

import pytest
from fastapi import status
from fastapi.testclient import TestClient

DEFAULT_PAGE_SIZE: int = 10
DEFAULT_PAGE_NUMBER: int = 1

# This should match whatever your seed function does
SEEDED_MODEL_COUNT = 8


@pytest.mark.asyncio
def test_list_models_pagination_non_default_params(
    client: TestClient,
) -> None:
    """Test pagination with non-default parameters, assuming DB is already seeded."""
    last_pagenum = math.ceil(SEEDED_MODEL_COUNT / DEFAULT_PAGE_SIZE)
    expected_last_page_items = SEEDED_MODEL_COUNT % DEFAULT_PAGE_SIZE or DEFAULT_PAGE_SIZE

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
def test_list_models_pagination_default_params(
    client: TestClient,
) -> None:
    """Test pagination with default parameters, assuming DB is already seeded."""
    last_pagenum = math.ceil(SEEDED_MODEL_COUNT / DEFAULT_PAGE_SIZE)
    expected_last_page_items = SEEDED_MODEL_COUNT % DEFAULT_PAGE_SIZE or DEFAULT_PAGE_SIZE

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
