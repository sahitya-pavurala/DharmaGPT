"""
Translation backend registry.

Default: sarvam (TRANSLATION_BACKEND in .env)
No fallback — if the configured backend fails, the exception propagates immediately.

Supported values:
  sarvam      — Sarvam translate API
  anthropic   — Claude via Anthropic API
  openai      — OpenAI chat translation
  ollama      — local Ollama model
  indictrans2 — local AI4Bharat IndicTrans2 model
  skip        — no-op, returns original text untouched (use for English-only sources)
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import requests
import structlog

log = structlog.get_logger()


@dataclass(frozen=True)
class TranslationResult:
    text: str
    backend: str
    source_lang: str
    target_lang: str
    skipped: bool = False


def _translate_sarvam(text: str, source_lang: str, target_lang: str, api_key: str) -> str:
    resp = requests.post(
        "https://api.sarvam.ai/translate",
        headers={"api-subscription-key": api_key, "Content-Type": "application/json"},
        json={
            "input": text,
            "source_language_code": source_lang,
            "target_language_code": target_lang,
            "speaker_gender": "Male",
            "mode": "formal",
            "enable_preprocessing": True,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return (resp.json().get("translated_text") or "").strip()


def _translate_anthropic(text: str, source_lang: str, target_lang: str, model: str, api_key: str) -> str:
    from anthropic import Anthropic
    client = Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": (
                f"Translate the following {source_lang} text to {target_lang}. "
                "Keep the translation faithful, preserve sacred names and terms. "
                "Return only the translation.\n\nText:\n" + text + "\n\nTranslation:"
            ),
        }],
    )
    return msg.content[0].text.strip()


class Translator:
    """Returned by get_translator(). Backend is pre-wired from config."""

    def __init__(self, backend: str, settings):
        self._backend = backend
        self._settings = settings

    @property
    def backend_name(self) -> str:
        return self._backend

    def translate(self, text: str, source_lang: str = "te", target_lang: str = "en") -> TranslationResult:
        s = self._settings
        backend = self._backend

        if backend == "skip":
            return TranslationResult(
                text=text, backend="skip",
                source_lang=source_lang, target_lang=target_lang, skipped=True,
            )

        from core.translation import TranslationBackend, TranslationConfig, translate_text

        outcome = translate_text(
            text,
            config=TranslationConfig(
                backend=TranslationBackend(backend),
                anthropic_model=s.anthropic_model,
                anthropic_api_key=s.anthropic_api_key,
                openai_api_key=s.openai_api_key,
                sarvam_api_key=s.sarvam_api_key,
                ollama_model=s.ollama_model,
                ollama_url=s.ollama_url,
                indictrans2_model=getattr(s, "indictrans2_model", "ai4bharat/indictrans2-indic-en-dist-200M"),
                backend_order=(backend,),
            ),
            source_lang=source_lang,
            target_lang=target_lang,
        )
        translated = outcome.text

        return TranslationResult(
            text=translated, backend=outcome.backend,
            source_lang=source_lang, target_lang=target_lang,
        )


@lru_cache(maxsize=1)
def get_translator() -> Translator:
    """Returns the configured Translator, cached for the process lifetime."""
    from core.config import get_settings
    settings = get_settings()
    backend = (settings.translation_backend or "sarvam").lower()
    log.info("translation_backend_loaded", backend=backend)
    return Translator(backend=backend, settings=settings)
