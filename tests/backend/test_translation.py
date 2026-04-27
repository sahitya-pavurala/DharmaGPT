from core.translation import TranslationBackend, TranslationConfig, translate_text


def test_translate_text_falls_back_on_rate_limit_to_ollama(monkeypatch):
    config = TranslationConfig(
        backend=TranslationBackend.auto,
        anthropic_api_key="test",
        anthropic_model="claude-test",
        ollama_model="qwen-test",
        ollama_url="http://localhost:11434",
        indictrans2_model="indic-test",
        local_first=False,
    )

    monkeypatch.setattr("core.translation._ollama_available", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        "core.translation._translate_with_anthropic",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("429 rate limit")),
    )
    monkeypatch.setattr("core.translation._translate_with_ollama", lambda *args, **kwargs: "ollama translation")

    outcome = translate_text("नमस्ते", config=config, source_lang="hi", target_lang="en")

    assert outcome.text == "ollama translation"
    assert outcome.backend == "ollama"
    assert outcome.version == "qwen-test"
    assert outcome.requested_mode == "auto"
    assert "anthropic" in outcome.attempted_backends
    assert "ollama" in outcome.attempted_backends


def test_translate_text_reaches_indictrans2_after_remote_rate_limits(monkeypatch):
    config = TranslationConfig(
        backend=TranslationBackend.auto,
        anthropic_api_key="test",
        anthropic_model="claude-test",
        ollama_model="qwen-test",
        ollama_url="http://localhost:11434",
        indictrans2_model="indic-test",
        local_first=False,
    )

    monkeypatch.setattr("core.translation._ollama_available", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        "core.translation._translate_with_anthropic",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("429 rate limit")),
    )
    monkeypatch.setattr(
        "core.translation._translate_with_ollama",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("429 rate limit")),
    )
    monkeypatch.setattr("core.translation._translate_with_indictrans2", lambda *args, **kwargs: "indic translation")

    outcome = translate_text("नमस्ते", config=config, source_lang="hi", target_lang="en")

    assert outcome.text == "indic translation"
    assert outcome.backend == "indictrans2"
    assert outcome.version == "indic-test"
    assert outcome.requested_mode == "auto"
    assert outcome.attempted_backends == ("anthropic", "ollama", "indictrans2")
