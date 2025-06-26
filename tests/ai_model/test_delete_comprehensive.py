# ruff: noqa: S101

"""Comprehensive tests for AI model DELETE endpoint filesystem and database cleanup."""

import shutil
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from vectorize.config import settings

_DELETE_ID = "2d2f3e4b-8c7f-4d2a-9f1e-0a6f3e4d2a5b"
_DELETE_TAG = "huge_model"
_NON_EXISTENT_ID = "12345678-1234-5678-1234-567812345678"


@pytest.mark.asyncio
@pytest.mark.ai_model
@pytest.mark.ai_model_delete
class TestDeleteAIModelComprehensive:
    """Comprehensive tests for the DELETE /models/{ai_model_id} endpoint."""

    @classmethod
    def _create_test_model_files(cls, model_tag: str) -> Path:
        """Create test model files on disk for testing deletion.
        
        Args:
            model_tag: The model tag to create files for
            
        Returns:
            Path to the created model directory
        """
        model_path = settings.model_upload_dir / model_tag
        model_path.mkdir(parents=True, exist_ok=True)

        # Create some test files
        (model_path / "config.json").write_text('{"model_type": "sentence-transformers"}')
        (model_path / "pytorch_model.bin").write_bytes(b"fake model weights")
        (model_path / "tokenizer.json").write_text('{"tokenizer": "test"}')

        # Create a subdirectory with files
        subdir = model_path / "tokenizer_config"
        subdir.mkdir(exist_ok=True)
        (subdir / "vocab.txt").write_text("test vocab")

        return model_path

    @classmethod
    def _verify_model_files_deleted(cls, model_tag: str) -> bool:
        """Verify that model files have been deleted from disk.
        
        Args:
            model_tag: The model tag to check
            
        Returns:
            True if files are deleted, False otherwise
        """
        model_path = settings.model_upload_dir / model_tag
        return not model_path.exists()

    @classmethod
    async def test_delete_with_filesystem_cleanup(cls, client: TestClient) -> None:
        """Test successful deletion of a model with filesystem cleanup."""
        # First, create test files for the model that will be deleted
        model_path = cls._create_test_model_files(_DELETE_TAG)
        assert model_path.exists(), "Test model files should exist before deletion"

        # Verify the model exists in the database
        get_response = client.get(f"/models/{_DELETE_TAG}")
        assert get_response.status_code == status.HTTP_200_OK
        model_data = get_response.json()
        assert model_data["model_tag"] == _DELETE_TAG

        # Get initial count of models
        response = client.get("/models?size=100")
        assert response.status_code == status.HTTP_200_OK
        initial_count = len(response.json()["items"])

        # Delete the model
        delete_response = client.delete(f"/models/{_DELETE_ID}")
        assert delete_response.status_code == status.HTTP_204_NO_CONTENT

        # Verify model is deleted from database
        get_response = client.get(f"/models/{_DELETE_TAG}")
        assert get_response.status_code == status.HTTP_404_NOT_FOUND

        # Verify count decreased
        response = client.get("/models?size=100")
        assert len(response.json()["items"]) == initial_count - 1

        # Verify files are deleted from filesystem
        assert cls._verify_model_files_deleted(_DELETE_TAG), "Model files should be deleted from disk"

    @classmethod
    async def test_delete_nonexistent_model(cls, client: TestClient) -> None:
        """Test deletion of a non-existent model."""
        # Get initial count
        response = client.get("/models?size=100")
        assert response.status_code == status.HTTP_200_OK
        initial_count = len(response.json()["items"])

        # Try to delete non-existent model
        delete_response = client.delete(f"/models/{_NON_EXISTENT_ID}")
        assert delete_response.status_code == status.HTTP_404_NOT_FOUND
        response_body = delete_response.json()
        assert response_body["code"] == "NOT_FOUND"

        # Verify count unchanged
        response = client.get("/models?size=100")
        assert len(response.json()["items"]) == initial_count

    @classmethod
    async def test_delete_model_without_filesystem_files(cls, client: TestClient) -> None:
        """Test deletion of a model that exists in DB but has no filesystem files."""
        # Create a test model in database only (no files)
        new_model_id = str(uuid4())
        new_model_tag = f"test_model_{new_model_id.replace('-', '_')}"

        # Create model via API (this won't create files)
        create_data = {
            "name": "Test Model for Deletion",
            "model_tag": new_model_tag,
            "source": "LOCAL"
        }
        create_response = client.post("/models", json=create_data)
        assert create_response.status_code == status.HTTP_201_CREATED
        created_model = create_response.json()
        created_id = created_model["id"]

        # Verify model exists in database but not on filesystem
        get_response = client.get(f"/models/{new_model_tag}")
        assert get_response.status_code == status.HTTP_200_OK

        model_path = settings.model_upload_dir / new_model_tag
        assert not model_path.exists(), "Model files should not exist on filesystem"

        # Get initial count
        response = client.get("/models?size=100")
        initial_count = len(response.json()["items"])

        # Delete the model
        delete_response = client.delete(f"/models/{created_id}")
        assert delete_response.status_code == status.HTTP_204_NO_CONTENT

        # Verify model is deleted from database
        get_response = client.get(f"/models/{new_model_tag}")
        assert get_response.status_code == status.HTTP_404_NOT_FOUND

        # Verify count decreased
        response = client.get("/models?size=100")
        assert len(response.json()["items"]) == initial_count - 1

    @classmethod
    async def test_delete_model_with_filesystem_permission_error(cls, client: TestClient) -> None:
        """Test deletion when filesystem deletion fails due to permissions."""
        # Create a test model with files
        test_model_tag = "permission_test_model"
        model_path = cls._create_test_model_files(test_model_tag)

        # Create model in database
        create_data = {
            "name": "Permission Test Model",
            "model_tag": test_model_tag,
            "source": "LOCAL"
        }
        create_response = client.post("/models", json=create_data)
        assert create_response.status_code == status.HTTP_201_CREATED
        created_model = create_response.json()
        created_id = created_model["id"]

        # Make the directory read-only to simulate permission error
        # Note: This might not work on all systems, but it's worth testing
        original_permissions = model_path.stat().st_mode
        try:
            model_path.chmod(0o444)  # Read-only

            # Attempt to delete the model
            delete_response = client.delete(f"/models/{created_id}")

            # The delete should still succeed in the database even if filesystem deletion fails
            # Our implementation logs the error but continues
            assert delete_response.status_code == status.HTTP_204_NO_CONTENT

            # Verify model is deleted from database
            get_response = client.get(f"/models/{test_model_tag}")
            assert get_response.status_code == status.HTTP_404_NOT_FOUND

        finally:
            # Restore permissions and clean up
            try:
                model_path.chmod(original_permissions)
                if model_path.exists():
                    shutil.rmtree(model_path)
            except Exception:
                pass  # Best effort cleanup

    @classmethod
    async def test_delete_model_with_complex_filesystem_structure(cls, client: TestClient) -> None:
        """Test deletion of a model with complex nested filesystem structure."""
        test_model_tag = "complex_structure_model"

        # Create model with complex structure
        model_path = settings.model_upload_dir / test_model_tag
        model_path.mkdir(parents=True, exist_ok=True)

        # Create multiple levels of directories and files
        (model_path / "config.json").write_text('{"model": "test"}')
        (model_path / "model.safetensors").write_bytes(b"safetensors data")

        # Create nested subdirectories
        tokenizer_dir = model_path / "tokenizer"
        tokenizer_dir.mkdir(exist_ok=True)
        (tokenizer_dir / "vocab.txt").write_text("vocab")
        (tokenizer_dir / "merges.txt").write_text("merges")

        deep_dir = model_path / "deep" / "nested" / "structure"
        deep_dir.mkdir(parents=True, exist_ok=True)
        (deep_dir / "deep_file.txt").write_text("deep content")

        # Create symlinks if supported
        try:
            symlink_path = model_path / "symlink_file.txt"
            symlink_path.symlink_to("config.json")
        except (OSError, NotImplementedError):
            # Symlinks not supported on this platform
            pass

        # Create model in database
        create_data = {
            "name": "Complex Structure Model",
            "model_tag": test_model_tag,
            "source": "LOCAL"
        }
        create_response = client.post("/models", json=create_data)
        assert create_response.status_code == status.HTTP_201_CREATED
        created_model = create_response.json()
        created_id = created_model["id"]

        # Verify complex structure exists
        assert model_path.exists()
        assert (model_path / "config.json").exists()
        assert (tokenizer_dir / "vocab.txt").exists()
        assert (deep_dir / "deep_file.txt").exists()

        # Delete the model
        delete_response = client.delete(f"/models/{created_id}")
        assert delete_response.status_code == status.HTTP_204_NO_CONTENT

        # Verify model is deleted from database
        get_response = client.get(f"/models/{test_model_tag}")
        assert get_response.status_code == status.HTTP_404_NOT_FOUND

        # Verify entire directory structure is deleted
        assert not model_path.exists(), "Entire model directory should be deleted"

    @classmethod
    async def test_delete_model_by_tag_vs_id(cls, client: TestClient) -> None:
        """Test that we can only delete by ID, not by tag (as per API design)."""
        # Try to delete using tag instead of ID
        delete_response = client.delete(f"/models/{_DELETE_TAG}")

        # This should result in a 404 because we're treating the tag as an ID
        assert delete_response.status_code == status.HTTP_404_NOT_FOUND

        # Verify the model still exists when accessed by tag
        get_response = client.get(f"/models/{_DELETE_TAG}")
        assert get_response.status_code == status.HTTP_200_OK

    @classmethod
    async def test_delete_idempotency(cls, client: TestClient) -> None:
        """Test that deleting the same model twice is handled gracefully."""
        # Create a test model for this test
        test_model_tag = "idempotency_test_model"
        cls._create_test_model_files(test_model_tag)

        create_data = {
            "name": "Idempotency Test Model",
            "model_tag": test_model_tag,
            "source": "LOCAL"
        }
        create_response = client.post("/models", json=create_data)
        assert create_response.status_code == status.HTTP_201_CREATED
        created_model = create_response.json()
        created_id = created_model["id"]

        # First deletion should succeed
        delete_response = client.delete(f"/models/{created_id}")
        assert delete_response.status_code == status.HTTP_204_NO_CONTENT

        # Second deletion should return 404
        delete_response2 = client.delete(f"/models/{created_id}")
        assert delete_response2.status_code == status.HTTP_404_NOT_FOUND

        # Verify files were deleted on first attempt
        assert cls._verify_model_files_deleted(test_model_tag)
