"""UUID validation utility for training module."""
import uuid

def is_valid_uuid(val: str) -> bool:
    try:
        uuid.UUID(str(val))
        return True
    except (ValueError, AttributeError, TypeError):
        return False

def normalize_uuid(val: str) -> str:
    """Gibt die UUID als 32-stelligen String ohne Bindestriche zurück (wie in DB gespeichert)."""
    try:
        return uuid.UUID(str(val)).hex
    except (ValueError, AttributeError, TypeError):
        return val
