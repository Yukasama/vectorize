"""Background service to handle GitHub and Huggingface upload processes."""

from txt2vec.upload.models import UploadTask


async def write_to_database(
    db: Annotated[AsyncSession, Depends(get_session)], upload_task: UploadTask
) -> dict:
    """"""

    db.add(upload_task)
    await db.commit()
    await db.refresh(upload_task)

    return {}
