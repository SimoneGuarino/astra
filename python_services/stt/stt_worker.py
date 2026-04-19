import json
import contextlib
import os
import sys
import traceback
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _bootstrap_cuda_dlls() -> None:
    """
    Make CUDA/cuDNN DLLs installed inside the venv visible to the current
    Python process before importing faster_whisper / ctranslate2.
    """
    exe_path = Path(sys.executable).resolve()
    venv_root = exe_path.parent.parent
    site_packages = venv_root / "Lib" / "site-packages"

    candidate_dirs = [
        site_packages / "nvidia" / "cublas" / "bin",
        site_packages / "nvidia" / "cudnn" / "bin",
        site_packages / "nvidia" / "cuda_nvrtc" / "bin",
        site_packages / "nvidia" / "cuda_runtime" / "bin",
    ]

    existing_dirs = [p for p in candidate_dirs if p.exists()]

    if existing_dirs:
        prepend = os.pathsep.join(str(p) for p in existing_dirs)
        os.environ["PATH"] = prepend + os.pathsep + os.environ.get("PATH", "")

        for dll_dir in existing_dirs:
            try:
                os.add_dll_directory(str(dll_dir))
            except (AttributeError, FileNotFoundError, OSError):
                pass


_bootstrap_cuda_dlls()

from faster_whisper import WhisperModel  # noqa: E402


MODEL_SIZE = os.environ.get("ASTRA_STT_MODEL", "small")
DEFAULT_LANGUAGE = os.environ.get("ASTRA_STT_LANGUAGE", "it")


def _create_model() -> WhisperModel:
    try:
        with contextlib.redirect_stdout(sys.stderr):
            return WhisperModel(
                MODEL_SIZE,
                device="cuda",
                compute_type="float16",
                cpu_threads=4,
            )
    except Exception as gpu_error:
        print(
            json.dumps(
                {
                    "warning": "GPU init failed, falling back to CPU",
                    "gpu_error": str(gpu_error),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )
        with contextlib.redirect_stdout(sys.stderr):
            return WhisperModel(
                MODEL_SIZE,
                device="cpu",
                compute_type="int8",
                cpu_threads=max(2, (os.cpu_count() or 4) // 2),
            )


MODEL = _create_model()


def transcribe(audio_path: str) -> dict[str, Any]:
    with contextlib.redirect_stdout(sys.stderr):
        segments, info = MODEL.transcribe(
            audio_path,
            language=DEFAULT_LANGUAGE,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 350,
                "speech_pad_ms": 120,
            },
            beam_size=1,
            best_of=1,
            temperature=0.0,
            patience=1.0,
            condition_on_previous_text=False,
            without_timestamps=True,
            compression_ratio_threshold=2.6,
            log_prob_threshold=-1.2,
            no_speech_threshold=0.45,
        )

    text = " ".join(
        segment.text.strip()
        for segment in segments
        if getattr(segment, "text", "").strip()
    ).strip()

    return {
        "ok": True,
        "audio_path": audio_path,
        "text": text,
        "language": getattr(info, "language", DEFAULT_LANGUAGE),
    }


def handle_request(request: dict[str, Any]) -> dict[str, Any]:
    audio_path = str(request["audio_path"])
    return transcribe(audio_path)


def safe_get(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key, "")
    return ""


def run_server() -> None:
    print("STT worker ready", file=sys.stderr, flush=True)

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        request: dict[str, Any] = {}
        try:
            request = json.loads(line)
            response = handle_request(request)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            response = {
                "ok": False,
                "audio_path": safe_get(locals().get("request"), "audio_path"),
                "error": f"{type(exc).__name__}: {exc}",
            }

        print(json.dumps(response, ensure_ascii=False), flush=True)


def main() -> None:
    try:
        print(json.dumps(transcribe(sys.argv[1]), ensure_ascii=False), flush=True)
    except Exception:
        print(
            traceback.format_exc(),
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--server":
        run_server()
    else:
        main()
