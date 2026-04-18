import json
import os
import sys
import traceback
from pathlib import Path


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

    print(
        json.dumps(
            {
                "debug": {
                    "venv_root": str(venv_root),
                    "existing_cuda_dirs": [str(p) for p in existing_dirs],
                }
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
        flush=True,
    )


_bootstrap_cuda_dlls()

from faster_whisper import WhisperModel  # noqa: E402


def _create_model() -> WhisperModel:
    # First try GPU, then fallback to CPU so Astra never completely breaks.
    try:
        return WhisperModel(
            "small",
            device="cuda",
            compute_type="float16",
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
        return WhisperModel(
            "small",
            device="cpu",
            compute_type="int8",
        )


MODEL = _create_model()


def transcribe(audio_path: str):
    segments, info = MODEL.transcribe(
        audio_path,
        language="it",
        vad_filter=True,
    )

    text = " ".join(
        segment.text.strip()
        for segment in segments
        if getattr(segment, "text", "").strip()
    ).strip()

    return {
        "text": text,
        "language": getattr(info, "language", "it"),
    }


def main():
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
    main()