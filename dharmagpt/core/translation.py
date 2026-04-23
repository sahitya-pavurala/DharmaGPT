from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Optional

import requests


class TranslationBackend(str, Enum):
    auto = "auto"
    anthropic = "anthropic"
    ollama = "ollama"
    indictrans2 = "indictrans2"
    skip = "skip"


@dataclass(frozen=True)
class TranslationConfig:
    backend: TranslationBackend = TranslationBackend.auto
    anthropic_model: str = "claude-3-5-sonnet-20241022"
    anthropic_api_key: str = ""
    ollama_model: str = "qwen2.5:7b"
    ollama_url: str = "http://localhost:11434"
    ollama_timeout_sec: int = 120
    indictrans2_model: str = "ai4bharat/indictrans2-indic-en-dist-200M"
    indictrans2_src_lang: str = "tel_Telu"
    indictrans2_tgt_lang: str = "eng_Latn"


def _normalize_flores_lang(lang: str) -> str:
    lang = (lang or "").strip()
    if lang in {"te", "tel", "telugu"}:
        return "tel_Telu"
    if lang in {"en", "eng", "english"}:
        return "eng_Latn"
    return lang


def _ollama_available(base_url: str, timeout_sec: int = 5) -> bool:
    try:
        resp = requests.get(base_url.rstrip("/") + "/api/tags", timeout=timeout_sec)
        resp.raise_for_status()
        return True
    except Exception:
        return False


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

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return tokenizer, model, device


def _split_sentences(text: str, lang_code: str) -> list[str]:
    try:
        from indicnlp.tokenize.sentence_tokenize import DELIM_PAT_NO_DANDA, sentence_split

        if lang_code == "eng_Latn":
            return [s.strip() for s in text.split(".") if s.strip()] or [text.strip()]
        sentences = sentence_split(text, lang=lang_code.split("_", 1)[0], delim_pat=DELIM_PAT_NO_DANDA)
        return [s.strip() for s in sentences if s.strip()] or [text.strip()]
    except Exception:
        return [text.strip()]


def _translate_with_indictrans2(
    text: str,
    config: TranslationConfig,
    source_lang: str,
    target_lang: str,
) -> str:
    import torch

    tokenizer, model, device = _load_indictrans2_model(config.indictrans2_model)
    sentences = _split_sentences(text, source_lang)
    outputs: list[str] = []

    with torch.no_grad():
        for sentence in sentences:
            inputs = tokenizer(
                sentence,
                truncation=True,
                padding=False,
                return_tensors="pt",
            ).to(device)
            generated = model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=5,
                num_return_sequences=1,
            )
            decoded = tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            if decoded:
                outputs.append(decoded[0].strip())

    return " ".join(piece for piece in outputs if piece).strip()


def translate_text(
    text: str,
    *,
    config: TranslationConfig,
    source_lang: str = "te",
    target_lang: str = "en",
) -> tuple[str, str]:
    """
    Translate text with a selectable backend.

    Returns a tuple of (translated_text, backend_used).
    """
    if config.backend == TranslationBackend.skip:
        return "[Translation skipped]", TranslationBackend.skip.value

    source_lang = _normalize_flores_lang(source_lang)
    target_lang = _normalize_flores_lang(target_lang)

    backend = TranslationBackend((config.backend.value if isinstance(config.backend, TranslationBackend) else str(config.backend)).lower())
    if backend == TranslationBackend.auto:
        if _ollama_available(config.ollama_url, timeout_sec=5):
            backend = TranslationBackend.ollama
        elif config.anthropic_api_key:
            backend = TranslationBackend.anthropic
        else:
            backend = TranslationBackend.indictrans2

    if backend == TranslationBackend.anthropic:
        return _translate_with_anthropic(text, config, source_lang, target_lang), backend.value

    if backend == TranslationBackend.ollama:
        return _translate_with_ollama(text, config, source_lang, target_lang), backend.value

    if backend == TranslationBackend.indictrans2:
        return _translate_with_indictrans2(text, config, source_lang, target_lang), backend.value

    raise ValueError(f"Unknown translation backend: {backend}")
