from __future__ import annotations

from python_services.stt import stt_worker


class FakeWhisperModel:
    calls: list[dict[str, object]] = []
    fail_cuda = False

    def __init__(self, model_size: str, *, device: str, compute_type: str, cpu_threads: int) -> None:
        self.calls.append(
            {
                "model_size": model_size,
                "device": device,
                "compute_type": compute_type,
                "cpu_threads": cpu_threads,
            }
        )
        if device == "cuda" and self.fail_cuda:
            raise RuntimeError("cuda unavailable")


def reset_fake_model() -> None:
    FakeWhisperModel.calls = []
    FakeWhisperModel.fail_cuda = False
    stt_worker.MODEL = None
    stt_worker.WhisperModel = FakeWhisperModel
    stt_worker.FASTER_WHISPER_IMPORT_ERROR = None
    stt_worker.DEFAULT_COMPUTE_TYPE = ""


def test_stt_device_cpu_initializes_cpu_path() -> None:
    reset_fake_model()
    stt_worker.DEFAULT_DEVICE = "cpu"

    stt_worker._create_model()

    assert FakeWhisperModel.calls[-1]["device"] == "cpu"
    assert FakeWhisperModel.calls[-1]["compute_type"] == "int8"


def test_stt_device_auto_falls_back_to_cpu_when_cuda_fails() -> None:
    reset_fake_model()
    stt_worker.DEFAULT_DEVICE = "auto"
    FakeWhisperModel.fail_cuda = True

    stt_worker._create_model()

    assert [call["device"] for call in FakeWhisperModel.calls] == ["cuda", "cpu"]
    assert FakeWhisperModel.calls[-1]["compute_type"] == "int8"


def test_stt_compute_type_env_override_is_respected() -> None:
    reset_fake_model()
    stt_worker.DEFAULT_DEVICE = "cpu"
    stt_worker.DEFAULT_COMPUTE_TYPE = "float32"

    stt_worker._create_model()

    assert FakeWhisperModel.calls[-1]["device"] == "cpu"
    assert FakeWhisperModel.calls[-1]["compute_type"] == "float32"


def test_stt_device_invalid_value_normalizes_to_auto() -> None:
    assert stt_worker._normalize_stt_device("bogus") == "auto"
    assert stt_worker._normalize_stt_device(" CUDA ") == "cuda"
