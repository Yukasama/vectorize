"""BAckground Task Service."""

import shutil
import tempfile
from pathlib import Path
from typing import cast

from git import Repo
from sqlmodel import select

from txt2vec.ai_model.models import AIModel
from txt2vec.common.status import TaskStatus
from txt2vec.upload.models import UploadTask

MODEL_DIR = Path("./models")
MODEL_DIR.mkdir(exist_ok=True)


def write_to_database(upload_id: str, git_repo: str, session_factory) -> None:
    """Process an upload task by cloning a Git repository, validating model files,
    copying them to a permanent location, and updating the database status.

    This function will:
      1. Mark the UploadTask as PENDING.
      2. Clone the specified GitHub repository (shallow clone).
      3. Validate presence of required '.bin' and '.json' files.
      4. Copy the repository contents to a permanent models directory.
      5. Create an AIModel record.
      6. Mark the UploadTask as COMPLETED or FAILED and record errors.

    Args:
        upload_id (str): Unique identifier of the UploadTask to process.
        git_repo (str): GitHub repository path in the form 'owner/repo'.
        session_factory (Callable[[], Session]): Factory function that returns a new database session.

    Raises:
        ValueError: If the repository does not contain any '.bin' or '.json' files.
        Exception: Propagates unexpected errors during Git operations, file system operations,
                   or database interactions.
    """

    with session_factory() as session:
        task: UploadTask = session.exec(
            select(UploadTask).where(UploadTask.id == upload_id)
        ).one()

        task.task_status = TaskStatus.PENDING
        session.commit()

    try:
        # 1. clone into tmp dir
        with tempfile.TemporaryDirectory() as tmpdir:
            Repo.clone_from(f"https://github.com/{git_repo}.git", tmpdir, depth=1)
            tmp_path = Path(tmpdir)

            # 2. validate
            bin_files = list(tmp_path.glob("*.bin"))
            json_files = list(tmp_path.glob("*.json"))
            if not bin_files or not json_files:
                raise ValueError("Repo must contain *.bin and *.json files")

            # 3. copy to permanent location
            tag = task.model_tag
            target_dir = MODEL_DIR / tag
            shutil.copytree(tmp_path, target_dir, dirs_exist_ok=True)

        # 4. create AIModel row
        with session_factory() as session:
            model = AIModel(tag=task.model_tag, local_path=str(target_dir))
            session.add(model)

            # 5. mark UploadTask done
            task = cast(
                UploadTask,
                session.exec(
                    select(UploadTask).where(UploadTask.id == upload_id)
                ).one(),
            )
            task.task_status = TaskStatus.COMPLETED
            session.commit()

    except Exception as exc:
        with session_factory() as session:
            task = session.exec(
                select(UploadTask).where(UploadTask.id == upload_id)
            ).one()
            task.task_status = TaskStatus.FAILED
            task.error_msg = str(exc)
            session.commit()

            # remove files if created
            maybe_dir = MODEL_DIR / task.model_tag
            if maybe_dir.exists():
                shutil.rmtree(maybe_dir, ignore_errors=True)
