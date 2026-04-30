from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
import threading
from typing import Iterable

import requests


_INDICTRANS2_LOCK = threading.Lock()


class TranslationBackend(str, Enum):
    auto = "auto"
    sarvam = "sarvam"
    anthropic = "anthropic"
    openai = "openai"
    ollama = "ollama"
    indictrans2 = "indictrans2"
    skip = "skip"


@dataclass(frozen=True)
class TranslationConfig:
    backend: TranslationBackend = TranslationBackend.auto
    anthropic_model: str = "claude-3-5-sonnet-20241022"
    anthropic_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"
    openai_api_key: str = ""
    sarvam_model: str = "sarvam-translate:v1"
    sarvam_api_key: str = ""
    ollama_model: str = "qwen2.5:7b"
    ollama_url: str = "http://localhost:11434"
    ollama_timeout_sec: int = 120
    indictrans2_model: str = "ai4bharat/indictrans2-indic-en-dist-200M"
    indictrans2_src_lang: str = "tel_Telu"
    indictrans2_tgt_lang: str = "eng_Latn"
    local_first: bool = True
    backend_order: tuple[str, ...] = ()


@dataclass(frozen=True)
class TranslationOutcome:
    text: str
    requested_mode: str
    backend: str
    version: str
    source_lang: str
    target_lang: str
    attempted_backends: tuple[str, ...] = ()
    fallback_reason: str | None = None

    @property
    def mode(self) -> str:
        return self.backend


def _backend_version(config: TranslationConfig, backend: TranslationBackend) -> str:
    if backend == TranslationBackend.sarvam:
        return config.sarvam_model
    if backend == TranslationBackend.anthropic:
        return config.anthropic_model
    if backend == TranslationBackend.openai:
        return config.openai_model
    if backend == TranslationBackend.ollama:
        return config.ollama_model
    if backend == TranslationBackend.indictrans2:
        return config.indictrans2_model
    return backend.value


def _normalize_backend(value: TranslationBackend | str) -> TranslationBackend:
    if isinstance(value, TranslationBackend):
        return value
    return TranslationBackend(str(value).lower())


def _normalize_flores_lang(lang: str) -> str:
    lang = (lang or "").strip()
    if lang in {"te", "tel", "telugu"}:
        return "tel_Telu"
    if lang in {"en", "eng", "english"}:
        return "eng_Latn"
    return lang


def _to_sarvam_lang(lang: str) -> str:
    lang = (lang or "").strip()
    if "-" in lang and len(lang) >= 5:
        return lang
    if lang in {"tel_Telu", "te", "tel", "telugu"}:
        return "te-IN"
    if lang in {"hin_Deva", "hi", "hin", "hindi"}:
        return "hi-IN"
    if lang in {"san_Deva", "sa", "san", "sanskrit"}:
        return "sa-IN"
    if lang in {"eng_Latn", "en", "eng", "english"}:
        return "en-IN"
    return lang


def _ollama_available(base_url: str, timeout_sec: int = 5) -> bool:
    try:
        resp = requests.get(base_url.rstrip("/") + "/api/tags", timeout=timeout_sec)
        resp.raise_for_status()
        return True
    except Exception:
        return False


def _is_rate_limit_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True

    response = getattr(exc, "response", None)
    if getattr(response, "status_code", None) == 429:
        return True

    text = str(exc).lower()
    return "rate limit" in text or "too many requests" in text or "429" in text


class TranslationProviderLimitReached(RuntimeError):
    pass


class TranslationBackendsExhausted(RuntimeError):
    def __init__(self, attempted_backends: Iterable[str], reason: str | None = None):
        self.attempted_backends = tuple(attempted_backends)
        self.reason = reason
        message = "No translation backend succeeded"
        if reason:
            message += f": {reason}"
        super().__init__(message)


_DISABLED_REMOTE_BACKENDS: set[TranslationBackend] = set()
_DISABLED_LOCK = threading.Lock()


def reset_translation_provider_state() -> None:
    with _DISABLED_LOCK:
        _DISABLED_REMOTE_BACKENDS.clear()


def _disable_remote_backend(backend: TranslationBackend) -> None:
    if backend in {TranslationBackend.sarvam, TranslationBackend.anthropic, TranslationBackend.openai}:
        with _DISABLED_LOCK:
            _DISABLED_REMOTE_BACKENDS.add(backend)


def _remote_backend_disabled(backend: TranslationBackend) -> bool:
    with _DISABLED_LOCK:
        return backend in _DISABLED_REMOTE_BACKENDS


def _backend_configured(backend: TranslationBackend, config: TranslationConfig) -> bool:
    if backend == TranslationBackend.sarvam:
        return bool(config.sarvam_api_key)
    if backend == TranslationBackend.anthropic:
        return bool(config.anthropic_api_key)
    if backend == TranslationBackend.openai:
        return bool(config.openai_api_key)
    return True


def _translate_with_sarvam(text: str, config: TranslationConfig, source_lang: str, target_lang: str) -> str:
    if not config.sarvam_api_key:
        raise RuntimeError("SARVAM_API_KEY is not set")

    resp = requests.post(
        "https://api.sarvam.ai/translate",
        headers={"api-subscription-key": config.sarvam_api_key},
        json={
            "input": text[:2000],
            "source_language_code": _to_sarvam_lang(source_lang),
            "target_language_code": _to_sarvam_lang(target_lang),
            "model": config.sarvam_model,
            "mode": "formal",
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return (data.get("translated_text") or "").strip()


def _translate_with_anthropic(text: str, config: TranslationConfig, source_lang: str, target_lang: str) -> str:
    if not config.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    from anthropic import Anthropic

    client = Anthropic(api_key=config.anthropic_api_key)
    message = client.messages.create(
        model=config.anthropic_model,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Translate the following {source_lang} text to {target_lang}. "
                    "Keep the translation faithful to the original meaning and tone. "
                    "Return only the translation.\n\n"
                    f"Text:\n{text}\n\nTranslation:"
                ),
            }
        ],
    )
    return message.content[0].text.strip()


def _translate_with_openai(text: str, config: TranslationConfig, source_lang: str, target_lang: str) -> str:
    if not config.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    from openai import OpenAI

    client = OpenAI(api_key=config.openai_api_key)
    response = client.chat.completions.create(
        model=config.openai_model,
        messages=[
            {
                "role": "system",
                "content": "Translate faithfully. Preserve sacred names and return only the translation.",
            },
            {
                "role": "user",
                "content": f"Translate this {source_lang} text to {target_lang}:\n\n{text}",
            },
        ],
        temperature=0.1,
    )
    return (response.choices[0].message.content or "").strip()


def _translate_with_ollama(text: str, config: TranslationConfig, source_lang: str, target_lang: str) -> str:
    endpoint = config.ollama_url.rstrip("/") + "/api/generate"
    payload = {
        "model": config.ollama_model,
        "stream": False,
        "prompt": (
            f"Translate {source_lang} to {target_lang} faithfully. "
            "Preserve names and sacred terms. Return only the translation.\n\n"
            f"Text:\n{text}\n\nTranslation:"
        ),
        "options": {"temperature": 0.2},
    }
    resp = requests.post(endpoint, json=payload, timeout=config.ollama_timeout_sec)
    resp.raise_for_status()
    data = resp.json()
    return (data.get("response") or "").strip()


@lru_cache(maxsize=4)
def _load_indictrans2_model(model_name: str):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from IndicTransToolkit import IndicProcessor

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    processor = IndicProcessor(inference=True)
    return tokenizer, model, device, processor


def _split_sentences(text: str, lang_code: str) -> list[str]:
    try:
        from indicnlp.tokenize.sentence_tokenize import DELIM_PAT_NO_DANDA, sentence_split

        if lang_code == "eng_Latn":
            return [s.strip() for s in text.split(".") if s.strip()] or [text.strip()]
        sentences = sentence_split(text, lang=lang_code.split("_", 1)[0], delim_pat=DELIM_PAT_NO_DANDA)
        return [s.strip() for s in sentences if s.strip()] or [text.strip()]
    except Exception:
        return [text.strip()]


def _split_long_translation_units(sentences: list[str], max_chars: int = 220) -> list[str]:
    units: list[str] = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= max_chars:
            units.append(sentence)
            continue

        words = sentence.split()
        current: list[str] = []
        current_len = 0
        for word in words:
            next_len = current_len + len(word) + (1 if current else 0)
            if current and next_len > max_chars:
                units.append(" ".join(current))
                current = [word]
                current_len = len(word)
            else:
                current.append(word)
                current_len = next_len
        if current:
            units.append(" ".join(current))
    return units or [" ".join(sentences).strip()]


def _translate_with_indictrans2(
    text: str,
    config: TranslationConfig,
    source_lang: str,
    target_lang: str,
) -> str:
    import torch

    with _INDICTRANS2_LOCK:
        tokenizer, model, device, processor = _load_indictrans2_model(config.indictrans2_model)
        sentences = _split_long_translation_units(_split_sentences(text, source_lang))
        outputs: list[str] = []

        with torch.no_grad():
            for start in range(0, len(sentences), 8):
                batch = processor.preprocess_batch(
                    sentences[start : start + 8],
                    src_lang=source_lang,
                    tgt_lang=target_lang,
                )
                inputs = tokenizer(
                    batch,
                    truncation=True,
                    padding="longest",
                    return_tensors="pt",
                    return_attention_mask=True,
                ).to(device)
                generated = model.generate(
                    **inputs,
                    use_cache=True,
                    min_length=0,
                    max_new_tokens=256,
                    num_beams=5,
                    num_return_sequences=1,
                )
                decoded = tokenizer.batch_decode(
                    generated,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                outputs.extend(
                    piece.strip()
                    for piece in processor.postprocess_batch(decoded, lang=target_lang)
                    if piece.strip()
                )

    return " ".join(piece for piece in outputs if piece).strip()


def _candidate_backends(
    requested_backend: TranslationBackend,
    *,
    local_first: bool = True,
    backend_order: tuple[str, ...] = (),
) -> list[TranslationBackend]:
    if backend_order:
        preferred = [_normalize_backend(item) for item in backend_order if item]
    elif local_first:
        preferred = [TranslationBackend.ollama, TranslationBackend.indictrans2, TranslationBackend.sarvam, TranslationBackend.anthropic, TranslationBackend.openai]
    else:
        preferred = [TranslationBackend.sarvam, TranslationBackend.anthropic, TranslationBackend.openai, TranslationBackend.ollama, TranslationBackend.indictrans2]
    if requested_backend == TranslationBackend.auto:
        return preferred

    candidates = [requested_backend]
    for backend in preferred:
        if backend not in candidates:
            candidates.append(backend)
    return candidates


def translate_text(
    text: str,
    *,
    config: TranslationConfig,
    source_lang: str = "te",
    target_lang: str = "en",
) -> TranslationOutcome:
    """
    Translate text with a selectable backend.
    """
    if config.backend == TranslationBackend.skip:
        return TranslationOutcome(
            text="[Translation skipped]",
            requested_mode=TranslationBackend.skip.value,
            backend=TranslationBackend.skip.value,
            version=TranslationBackend.skip.value,
            source_lang=source_lang,
            target_lang=target_lang,
        )

    source_lang = _normalize_flores_lang(source_lang)
    target_lang = _normalize_flores_lang(target_lang)

    requested_backend = _normalize_backend(config.backend)
    candidates = _candidate_backends(
        requested_backend,
        local_first=getattr(config, "local_first", True),
        backend_order=getattr(config, "backend_order", ()),
    )
    attempted: list[str] = []
    fallback_reason: str | None = None

    for backend in candidates:
        if not _backend_configured(backend, config):
            continue
        if _remote_backend_disabled(backend):
            attempted.append(backend.value)
            fallback_reason = f"rate_limited:{backend.value}"
            continue
        attempted.append(backend.value)
        try:
            if backend == TranslationBackend.sarvam:
                translated = _translate_with_sarvam(text, config, source_lang, target_lang)
            elif backend == TranslationBackend.anthropic:
                translated = _translate_with_anthropic(text, config, source_lang, target_lang)
            elif backend == TranslationBackend.openai:
                translated = _translate_with_openai(text, config, source_lang, target_lang)
            elif backend == TranslationBackend.ollama:
                if backend == TranslationBackend.ollama and not _ollama_available(config.ollama_url, timeout_sec=5):
                    raise RuntimeError("Ollama is unavailable")
                translated = _translate_with_ollama(text, config, source_lang, target_lang)
            elif backend == TranslationBackend.indictrans2:
                translated = _translate_with_indictrans2(text, config, source_lang, target_lang)
            else:
                raise ValueError(f"Unknown translation backend: {backend}")

            return TranslationOutcome(
                text=translated,
                requested_mode=requested_backend.value,
                backend=backend.value,
                version=_backend_version(config, backend),
                source_lang=source_lang,
                target_lang=target_lang,
                attempted_backends=tuple(attempted),
                fallback_reason=fallback_reason,
            )
        except Exception as exc:
            is_rate_limited = _is_rate_limit_error(exc)
            if is_rate_limited:
                _disable_remote_backend(backend)
                fallback_reason = f"rate_limited:{backend.value}"
                continue
            if backend != TranslationBackend.indictrans2:
                fallback_reason = f"failed:{backend.value}"
                continue
            raise

    if TranslationBackend.indictrans2 not in candidates:
        translated = _translate_with_indictrans2(text, config, source_lang, target_lang)
        return TranslationOutcome(
            text=translated,
            requested_mode=requested_backend.value,
            backend=TranslationBackend.indictrans2.value,
            version=_backend_version(config, TranslationBackend.indictrans2),
            source_lang=source_lang,
            target_lang=target_lang,
            attempted_backends=tuple(attempted + [TranslationBackend.indictrans2.value]),
            fallback_reason=fallback_reason,
        )

    raise TranslationBackendsExhausted(attempted, fallback_reason)


async def translate_text_async(
    text: str,
    *,
    config: TranslationConfig,
    source_lang: str = "te",
    target_lang: str = "en",
) -> TranslationOutcome:
    import asyncio

    return await asyncio.to_thread(
        translate_text,
        text,
        config=config,
        source_lang=source_lang,
        target_lang=target_lang,
    )
