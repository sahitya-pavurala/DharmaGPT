"""
core/backends — pluggable provider registry for DharmaGPT.

Each capability is resolved once at startup from .env settings and cached.
Callers import from here; they never reference a concrete provider directly.

    from core.backends import get_translator, get_embedder, get_llm
"""
from core.backends.translation import get_translator, TranslationResult
from core.backends.embedding import get_embedder
from core.backends.llm import get_llm, invoke_chat_model, ainvoke_chat_model

__all__ = [
    "get_translator",
    "TranslationResult",
    "get_embedder",
    "get_llm",
    "invoke_chat_model",
    "ainvoke_chat_model",
]
