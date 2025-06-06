"""Check if the feature names match any of the allowed schemas."""

__all__ = ["_match_schema"]

_ALLOWED_SCHEMAS: list[set[str]] = [
    {"prompt", "chosen", "rejected"},
    {"prompt", "chosen"},
    {"question", "positive"},
    {"question", "answer"},
    {"question", "positive", "negative"},
    {"question", "chosen", "rejected"},
    {"input", "chosen", "rejected"},
]


def _match_schema(feature_names: set[str]) -> bool:
    """Check if feature_names matches any allowed schema."""
    return any(set(schema).issubset(feature_names) for schema in _ALLOWED_SCHEMAS)
