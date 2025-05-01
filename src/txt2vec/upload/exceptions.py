"""Custom-Fehler für Upload-Prozesse."""

from fastapi import status

from txt2vec.errors import AppError, ErrorCode



__all__ = ["InvalidModelError"]


class InvalidModelError(AppError):
    """Fehler beim Laden eines Modells."""

    error_code = ErrorCode.INVALID_FILE
    message = (
        "Das Modell konnte nicht geladen werden. "
        "Bitte überprüfe model_id und tag."
    )
    status_code = status.HTTP_400_BAD_REQUEST

class DatabaseError(AppError):
    """Fehler beim Zugriff auf die Datenbank."""

    error_code = ErrorCode.DATABASE_ERROR
    message = "Datenbankfehler. Bitte überprüfe die Verbindung."
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
