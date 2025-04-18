from fastapi import APIRouter
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
from loguru import logger

from txt2vec.handle_exceptions import AppError, handle_exceptions


router = APIRouter(tags=["Model Upload"])


class ModelUploadRequest(BaseModel):
    model_tag: str


@router.get("/")
def helloworld() -> dict[str, str]:
    return {"message": "Hello World!"}


@router.post("/tags")
@handle_exceptions
async def upload_model(request: ModelUploadRequest) -> dict[str, str]:
    model_tag = request.model_tag
    save_path = Path("data/uploads") / model_tag.replace("/", "__")
    save_path.mkdir(parents=True, exist_ok=True)

    logger.info("Received model upload request: model_tag='{}'", model_tag)

    try:
        logger.debug("Lade Modell von Hugging Face: '{}'", model_tag)
        model = AutoModel.from_pretrained(model_tag)
        tokenizer = AutoTokenizer.from_pretrained(model_tag)

        logger.debug("Speichere Modell und Tokenizer zu '{}'", save_path)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

    except Exception as e:
        logger.exception("Fehler beim Upload von Modell '{}'", model_tag)
        raise AppError(f"Failed to upload model '{model_tag}': {str(e)}") from e

    logger.success("Model '{}' uploaded successfully!", model_tag)
    return {"message": f"Model '{model_tag}' uploaded successfully!"}
