"""Check if the feature names match any of the allowed schemas."""

from vectorize.config.config import settings

__all__ = ["_match_schema"]


def _match_schema(feature_names: set[str]) -> bool:
    """Check if feature_names matches any allowed schema.

    Args:
        feature_names: Set of column names from the dataset

    Returns:
        True if the dataset matches any allowed schema, False otherwise
    """
    for schema in settings.dataset_hf_allowed_schemas:
        required_fields = set(schema) if isinstance(schema, (list, tuple)) else {schema}
        if required_fields.issubset(feature_names):
            return True

    return False
