"""Service zum Laden und Speichern von Hugging Face Modellen.

Dieses Modul lädt Modelle von Hugging Face, cached sie lokal und speichert sie
in der Datenbank, falls sie noch nicht vorhanden sind.
"""

from huggingface_hub import snapshot_download
from loguru import logger
from sqlmodel.ext.asyncio.session import AsyncSession
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from txt2vec.ai_model import AIModel
from txt2vec.ai_model.models import ModelSource
from txt2vec.ai_model.repository import get_ai_model, save_ai_model
from txt2vec.upload.exceptions import DatabaseError, InvalidModelError

_models = {}


async def load_model_and_save_to_db(
    model_id: str, tag: str, db: AsyncSession
) -> None:
    """Lädt ein Hugging Face Modell, cached es lokal und speichert es in die Datenbank.

    Args:
        model_id (str): Die ID des Modells auf Hugging Face.
        tag (str): Der Tag oder die Version des Modells.
        db (AsyncSession): Die Datenbank-Session.

    Raises:
        InvalidModelError: Wenn das Modell nicht geladen werden konnte.
        DatabaseError: Wenn ein Fehler beim Zugriff auf die Datenbank auftritt.
    """
    key = f"{model_id}@{tag}"

    if key not in _models:
        try:
            logger.info(f"Lade Modell '{key}'...")
            snapshot_path = snapshot_download(
                repo_id=model_id,
                revision=tag,
                cache_dir="./hf_cache",
            )
            tokenizer = AutoTokenizer.from_pretrained(snapshot_path)
            model = AutoModelForSequenceClassification.from_pretrained(snapshot_path)
            _models[key] = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
            )
            logger.info(f"Modell '{key}' erfolgreich geladen und gecached.")
        except Exception as e:
            logger.exception(f"Fehler beim Laden des Modells '{key}'")
            raise InvalidModelError from e
    else:
        logger.info(f"Modell '{key}' bereits im Cache.")

    # In die Datenbank schreiben, wenn noch nicht vorhanden
    try:
        try:
            await get_ai_model(db, key)  # Wenn gefunden → nichts tun
            logger.info(f"Modell '{key}' bereits in der Datenbank.")
            return
        except Exception:
            logger.info(
                f"Modell '{key}' nicht in der DB - wird gespeichert..."
            )

        ai_model = AIModel(
            model_tag=key,
            name=model_id,
            source=ModelSource.HUGGINGFACE,
        )
        await save_ai_model(db, ai_model)
        logger.info(f"Modell '{key}' erfolgreich in der Datenbank gespeichert.")

    except Exception as db_error:
        logger.error(f"Fehler beim Datenbankzugriff für Modell '{key}': {db_error}")
        raise DatabaseError from db_error
