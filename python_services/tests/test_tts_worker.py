from __future__ import annotations

import pytest

from python_services.tts import tts_worker


def test_sanitize_tts_input_removes_surrogates() -> None:
    sanitized = tts_worker.sanitize_tts_input("ciao\ud800 mondo")

    assert "\ud800" not in sanitized
    sanitized.encode("utf-8", "strict")


def test_sanitize_tts_input_converts_non_string() -> None:
    assert tts_worker.sanitize_tts_input(1234) == "1234"


def test_sanitize_tts_input_removes_null_and_control_chars() -> None:
    sanitized = tts_worker.sanitize_tts_input("ciao\x00\x01 mondo")

    assert sanitized == "ciao mondo"


def test_prepare_worker_tts_text_never_returns_surrogates() -> None:
    prepared = tts_worker.prepare_worker_tts_text("testo valido\udfff con controllo\x00")

    assert "\udfff" not in prepared
    prepared.encode("utf-8", "strict")


def test_chatterbox_generation_validates_before_model_call() -> None:
    engine = object.__new__(tts_worker.ChatterboxMultilingualEngine)

    class Model:
        called = False

        def generate(self, *_args, **_kwargs):
            self.called = True
            return []

    model = Model()
    engine.model = model
    engine.preprocessor = tts_worker.TextPreprocessor()
    engine.device = "cpu"
    engine.sample_rate = tts_worker.SAMPLE_RATE

    with pytest.raises(tts_worker.ControlledTtsInputError):
        engine._generate_one_part("\ud800", {})

    assert not model.called


def test_kokoro_fallback_receives_sanitized_text_only() -> None:
    engine = object.__new__(tts_worker.TtsEngine)

    class Primary:
        engine_name = "chatterbox_multilingual"

        def synthesize(self, **_kwargs):
            raise RuntimeError("primary failed")

    class Fallback:
        received = ""

        def synthesize(self, *, text, output_path, voice, speed):
            self.received = text
            return tts_worker.SynthesisResult(
                output_path=output_path,
                normalized_text=text,
                sample_rate=tts_worker.SAMPLE_RATE,
                engine="kokoro",
            )

    fallback = Fallback()
    engine.engine = Primary()
    engine.device = "cpu"
    engine._get_kokoro_fallback = lambda: fallback
    engine._empty_cuda_cache_on_failure = lambda: None

    result = engine.synthesize("ciao\ud800 mondo", "out.wav")

    assert result.engine == "kokoro"
    assert fallback.received == "ciao mondo"
    fallback.received.encode("utf-8", "strict")


def test_invalid_text_produces_controlled_error() -> None:
    with pytest.raises(tts_worker.ControlledTtsInputError):
        tts_worker.validate_speakable_text("\ud800\x00", "test")
