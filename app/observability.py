from functools import lru_cache

from langfuse import Langfuse

from app.secrets_utils import get_secrets


@lru_cache
def get_langfuse() -> Langfuse | None:
    secrets = get_secrets()
    if not secrets["LANGFUSE_PUBLIC_KEY"] or not secrets["LANGFUSE_SECRET_KEY"]:
        return None
    return Langfuse(
        public_key=secrets["LANGFUSE_PUBLIC_KEY"],
        secret_key=secrets["LANGFUSE_SECRET_KEY"],
        host=secrets["LANGFUSE_HOST"],
    )
