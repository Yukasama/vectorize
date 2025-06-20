"""Actions service."""

from vectorize.actions.repository import get_actions_db


def get_actions_svc() -> str:
    """Get actions service status."""
    actions_db = get_actions_db()
