from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum

import requests


class LLMBackend(str, Enum):
    anthropic = "anthropic"
    openai = "openai"
    ollama = "ollama"


@dataclass(frozen=True)
class LLMConfig:
    backend: LLMBackend
    model: str
    api_key: str = ""
    base_url: str = "http://localhost:11434"
    timeout_sec: int = 120
    max_tokens: int = 1024


def generate_text_sync(system: str, messages: list[dict], config: LLMConfig) -> str:
    """
    Generate a chat-style response using a selectable backend.
    """
    if config.backend == LLMBackend.anthropic:
        from anthropic import Anthropic

        if not config.api_key:
            raise RuntimeError("Anthropic backend selected but no API key was provided")

        client = Anthropic(api_key=config.api_key)
        response = client.messages.create(
            model=config.model,
            max_tokens=config.max_tokens,
            system=system,
            messages=messages,
        )
        return response.content[0].text

    if config.backend == LLMBackend.openai:
        from openai import OpenAI

        client = OpenAI(api_key=config.api_key or "EMPTY", base_url=config.base_url)
        response = client.chat.completions.create(
            model=config.model,
            messages=[{"role": "system", "content": system}, *messages],
            max_tokens=config.max_tokens,
        )
        return (response.choices[0].message.content or "").strip()

    if config.backend == LLMBackend.ollama:
        endpoint = config.base_url.rstrip("/") + "/api/chat"
        payload = {
            "model": config.model,
            "stream": False,
            "messages": [{"role": "system", "content": system}, *messages],
            "options": {"temperature": 0.2, "num_predict": config.max_tokens},
        }
        resp = requests.post(endpoint, json=payload, timeout=config.timeout_sec)
        resp.raise_for_status()
        data = resp.json()
        return ((data.get("message") or {}).get("content") or "").strip()

    raise ValueError(f"Unsupported LLM backend: {config.backend}")


async def generate_text_async(system: str, messages: list[dict], config: LLMConfig) -> str:
    return await asyncio.to_thread(generate_text_sync, system, messages, config)
