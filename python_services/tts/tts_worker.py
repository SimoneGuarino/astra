from __future__ import annotations

import json
import logging
import os
import re
import sys
import traceback
import unicodedata
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal
from python_services.config_loader import get_config

warnings.filterwarnings(
    "ignore",
    message=r"dropout option adds dropout after all but last recurrent layer.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"`torch\.nn\.utils\.weight_norm` is deprecated.*",
    category=FutureWarning,
)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import numpy as np
import soundfile as sf

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from kokoro import KPipeline


SAMPLE_RATE = 24000
KOKORO_REPO_ID = os.environ.get(
    "ASTRA_TTS_REPO_ID",
    get_config("tts.repo_id", "hexgrad/Kokoro-82M"),
)

DEFAULT_VOICE = os.environ.get(
    "ASTRA_TTS_VOICE",
    get_config("tts.voice", "if_sara"),
)

DEFAULT_SPEED = float(os.environ.get(
    "ASTRA_TTS_SPEED",
    str(get_config("tts.speed", 0.96)),
))

DEFAULT_ENGINE = os.environ.get(
    "ASTRA_TTS_ENGINE",
    get_config("tts.engine", "auto"),
).strip().lower()

DEFAULT_DEVICE = os.environ.get(
    "ASTRA_TTS_DEVICE",
    get_config("tts.device", "auto"),
).strip().lower()

DEFAULT_AUDIO_PROMPT_PATH = os.environ.get(
    "ASTRA_TTS_AUDIO_PROMPT_PATH",
    get_config("tts.audio_prompt_path", ""),
).strip()

DEFAULT_LANGUAGE_ID = os.environ.get(
    "ASTRA_TTS_LANGUAGE_ID",
    get_config("tts.language_id", "it"),
).strip().lower() or "it"

DEFAULT_CFG_WEIGHT = float(os.environ.get(
    "ASTRA_TTS_CFG_WEIGHT",
    str(get_config("tts.cfg_weight", 0.45)),
))

DEFAULT_EXAGGERATION = float(os.environ.get(
    "ASTRA_TTS_EXAGGERATION",
    str(get_config("tts.exaggeration", 0.55)),
))

DEFAULT_TIMEOUT_SECS = int(os.environ.get(
    "ASTRA_TTS_TIMEOUT_SECS",
    str(get_config("tts.timeout_secs", 240)),
))

CHATTERBOX_MAX_CHARS = int(os.environ.get(
    "ASTRA_CHATTERBOX_MAX_CHARS",
    str(get_config("tts.chatterbox_max_chars", 110)),
))

CHATTERBOX_RETRY_MAX_CHARS = int(os.environ.get(
    "ASTRA_CHATTERBOX_RETRY_MAX_CHARS",
    str(get_config("tts.chatterbox_retry_max_chars", 62)),
))

EngineName = Literal["auto", "kokoro", "chatterbox_multilingual", "chatterbox_turbo"]
PronunciationMode = Literal["plain", "kokoro"]


@dataclass(frozen=True)
class VoiceProfile:
    acknowledgement_pause_ms: int = 90
    short_reply_pause_ms: int = 110
    sentence_pause_ms: int = 130
    long_sentence_pause_ms: int = 175
    comma_pause_ms: int = 85
    semicolon_pause_ms: int = 150
    colon_pause_ms: int = 170
    question_pause_ms: int = 220
    exclamation_pause_ms: int = 185
    list_item_pause_ms: int = 190
    short_ack_speed: float = 0.94
    short_reply_speed: float = 0.955
    short_question_speed: float = 0.95
    long_explanation_speed: float = 0.97


@dataclass(frozen=True)
class SynthesisResult:
    output_path: str
    normalized_text: str
    sample_rate: int
    engine: str


class ItalianNumberNormalizer:
    _units = [
        "zero",
        "uno",
        "due",
        "tre",
        "quattro",
        "cinque",
        "sei",
        "sette",
        "otto",
        "nove",
        "dieci",
        "undici",
        "dodici",
        "tredici",
        "quattordici",
        "quindici",
        "sedici",
        "diciassette",
        "diciotto",
        "diciannove",
    ]
    _tens = {
        20: "venti",
        30: "trenta",
        40: "quaranta",
        50: "cinquanta",
        60: "sessanta",
        70: "settanta",
        80: "ottanta",
        90: "novanta",
    }

    def normalize(self, text: str) -> str:
        return re.sub(r"\b\d{1,6}\b", lambda match: self._to_words(int(match.group(0))), text)

    def _to_words(self, value: int) -> str:
        if value < 20:
            return self._units[value]
        if value < 100:
            tens_value = value // 10 * 10
            unit = value % 10
            tens = self._tens[tens_value]
            if unit in (1, 8):
                tens = tens[:-1]
            return tens if unit == 0 else f"{tens}{self._units[unit]}"
        if value < 1000:
            hundreds = value // 100
            rest = value % 100
            prefix = "cento" if hundreds == 1 else f"{self._units[hundreds]}cento"
            if 80 <= rest < 90:
                prefix = prefix[:-1]
            return prefix if rest == 0 else f"{prefix}{self._to_words(rest)}"
        if value < 1_000_000:
            thousands = value // 1000
            rest = value % 1000
            prefix = "mille" if thousands == 1 else f"{self._to_words(thousands)}mila"
            return prefix if rest == 0 else f"{prefix} {self._to_words(rest)}"
        return str(value)


class TextPreprocessor:
    _apostrophe_chars = "’`´ʻʼʽˈ"
    _code_block_pattern = re.compile(r"```.*?```", re.DOTALL)
    _inline_code_pattern = re.compile(r"`([^`]{1,80})`")
    _url_pattern = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
    _markdown_link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    _bullet_pattern = re.compile(r"(?m)^\s*[-*+]\s+")
    _numbered_bullet_pattern = re.compile(r"(?m)^\s*\d+[.)]\s+")
    _acronym_pattern = re.compile(r"\b[A-Z]{2,6}\b")
    _symbol_replacements = {
        "%": " per cento ",
        "+": " piu ",
        "=": " uguale ",
        "@": " chiocciola ",
        "|": " oppure ",
    }
    _acknowledgement_pattern = re.compile(
        r"^(certo|ok|okay|va bene|perfetto|capito|chiaro|bene|sì|si|assolutamente|dimmi|eccomi|ci sono|un attimo)[.!]?$",
        re.IGNORECASE,
    )
    _conversational_prefix_pattern = re.compile(
        r"^(certo|ok|okay|va bene|perfetto|capito|chiaro|bene|sì|si|dimmi|eccomi|allora),\s+",
        re.IGNORECASE,
    )
    _letter_names = {
        "A": "a",
        "B": "bi",
        "C": "ci",
        "D": "di",
        "E": "e",
        "F": "effe",
        "G": "gi",
        "H": "acca",
        "I": "i",
        "J": "i lunga",
        "K": "kappa",
        "L": "elle",
        "M": "emme",
        "N": "enne",
        "O": "o",
        "P": "pi",
        "Q": "cu",
        "R": "erre",
        "S": "esse",
        "T": "ti",
        "U": "u",
        "V": "vi",
        "W": "doppia vi",
        "X": "ics",
        "Y": "ipsilon",
        "Z": "zeta",
    }
    _technical_terms = {
        "LLM": "elle elle emme",
        "TTS": "ti ti esse",
        "STT": "esse ti ti",
        "UI": "u ai",
        "UX": "u x",
        "JSON": "geison",
        "HTTP": "acca ti ti pi",
        "API": "api",
        "SQL": "esse cu elle",
        "CPU": "ci pi u",
        "GPU": "gi pi u",
        "RAM": "ram",
    }
    _kokoro_pronunciation_overrides = [
        (re.compile(r"(?i)\bè\b"), "[è](/ˈɛ/)"),
        (re.compile(r"(?i)\bcioè\b"), "[cioè](/tʃoˈɛ/)"),
        (re.compile(r"(?i)\bperché\b"), "[perché](/perˈke/)"),
        (re.compile(r"(?i)\bpoiché\b"), "[poiché](/poiˈke/)"),
        (re.compile(r"(?i)\bné\b"), "[né](/ˈne/)"),
        (re.compile(r"(?i)\baffinché\b"), "[affinché](/affinˈke/)"),
        (re.compile(r"(?i)\bgiacché\b"), "[giacché](/dʒakˈke/)"),
    ]

    def __init__(self, profile: VoiceProfile | None = None) -> None:
        self.profile = profile or VoiceProfile()
        self.number_normalizer = ItalianNumberNormalizer()

    def normalize(self, text: str, pronunciation_mode: PronunciationMode = "plain") -> str:
        text = self._normalize_unicode(text)
        text = text.strip()
        if not text:
            return ""

        text = self._markdown_link_pattern.sub(r"\1", text)
        text = self._code_block_pattern.sub(" blocco di codice omesso. ", text)
        text = self._inline_code_pattern.sub(lambda match: self._speak_inline_code(match.group(1)), text)
        text = self._url_pattern.sub(" link disponibile ", text)

        text = text.replace("->", " porta a ")
        text = text.replace("=>", " restituisce ")
        text = text.replace("&", " e ")

        text = self._bullet_pattern.sub("Punto. ", text)
        text = self._numbered_bullet_pattern.sub("Punto. ", text)
        for source, replacement in self._symbol_replacements.items():
            text = text.replace(source, replacement)
        text = re.sub(r"[*_#>]+", " ", text)

        for source, replacement in self._technical_terms.items():
            text = re.sub(rf"\b{re.escape(source)}\b", replacement, text)

        text = self._acronym_pattern.sub(lambda match: self._spell_acronym(match.group(0)), text)
        text = self.number_normalizer.normalize(text)
        text = self._normalize_punctuation(text)
        text = self._shape_conversational_prefixes(text)
        if pronunciation_mode == "kokoro":
            text = self._apply_kokoro_pronunciation_overrides(text)
        text = re.sub(r"\s+", " ", text).strip()

        if text and text[-1] not in ".!?;:":
            text += "."

        return text

    def pause_ms(self, normalized_text: str) -> int:
        stripped = normalized_text.rstrip()
        if not stripped:
            return 80
        if self._acknowledgement_pattern.match(stripped):
            return self.profile.acknowledgement_pause_ms
        if self._is_short_reply(stripped):
            return self.profile.short_reply_pause_ms
        if stripped.endswith("?"):
            return self.profile.question_pause_ms
        if stripped.endswith("!"):
            return self.profile.exclamation_pause_ms
        if stripped.endswith(":"):
            return self.profile.colon_pause_ms
        if stripped.endswith(";"):
            return self.profile.semicolon_pause_ms
        if stripped.startswith("Punto."):
            return self.profile.list_item_pause_ms
        if stripped.endswith(","):
            return self.profile.comma_pause_ms
        if len(stripped) > 180:
            return self.profile.long_sentence_pause_ms
        return self.profile.sentence_pause_ms

    def speaking_speed(self, normalized_text: str, requested_speed: float) -> float:
        stripped = normalized_text.strip()
        if self._acknowledgement_pattern.match(stripped):
            return min(requested_speed, self.profile.short_ack_speed)
        if stripped.endswith("?") and len(stripped) <= 72:
            return min(requested_speed, self.profile.short_question_speed)
        if self._is_short_reply(stripped):
            return min(requested_speed, self.profile.short_reply_speed)
        if len(stripped) > 220:
            return max(requested_speed, self.profile.long_explanation_speed)
        return requested_speed

    def _normalize_unicode(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        text = text.replace("\xa0", " ")
        for ch in self._apostrophe_chars:
            text = text.replace(ch, "'")
        text = text.replace("“", '"').replace("”", '"')
        return text

    def _apply_kokoro_pronunciation_overrides(self, text: str) -> str:
        for pattern, replacement in self._kokoro_pronunciation_overrides:
            text = pattern.sub(replacement, text)
        return text

    def _normalize_punctuation(self, text: str) -> str:
        text = text.replace("\u2026", "...")
        text = text.replace("—", ", ")
        text = text.replace("–", ", ")
        text = re.sub(r"\.{4,}", "...", text)
        text = text.replace("...", ". ")
        text = re.sub(r"([!?]){2,}", r"\1", text)
        text = re.sub(r"\(([^)]+)\)", r", \1,", text)
        text = re.sub(r"\[([^\]]+)\]", r", \1,", text)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = re.sub(r"([,.;:!?])([^\s,.;:!?])", r"\1 \2", text)
        text = re.sub(r"\s+-\s+", ", ", text)
        text = re.sub(r"\s*/\s*", " o ", text)
        text = re.sub(r",\s*,+", ", ", text)
        return text

    def _shape_conversational_prefixes(self, text: str) -> str:
        return self._conversational_prefix_pattern.sub(lambda match: f"{match.group(1).capitalize()}, ", text)

    def _is_short_reply(self, text: str) -> bool:
        return len(text) <= 70 and len(re.findall(r"\w+", text)) <= 9

    def _spell_acronym(self, value: str) -> str:
        if value in self._technical_terms:
            return self._technical_terms[value]
        return " ".join(self._letter_names.get(letter, letter.lower()) for letter in value)

    def _speak_inline_code(self, value: str) -> str:
        value = value.strip()
        if not value:
            return " "
        if len(value) <= 24 and re.fullmatch(r"[A-Za-z0-9_.:-]+", value):
            return f" {value.replace('_', ' ')} "
        return " frammento tecnico "


def select_device(device: str) -> str:
    requested = (device or "auto").strip().lower()

    if requested not in {"auto", "cpu", "cuda"}:
        return "cpu"

    if requested == "cpu":
        return "cpu"

    if requested == "cuda":
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    # auto
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


@lru_cache(maxsize=1)
def import_chatterbox_modules() -> tuple[Any, Any]:
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    return ChatterboxMultilingualTTS, ChatterboxTurboTTS


class BaseEngine:
    engine_name: str
    pronunciation_mode: PronunciationMode

    def synthesize(self, text: str, output_path: str, voice: str, speed: float) -> SynthesisResult:
        raise NotImplementedError


class KokoroEngine(BaseEngine):
    engine_name = "kokoro"
    pronunciation_mode: PronunciationMode = "kokoro"

    def __init__(self, preprocessor: TextPreprocessor) -> None:
        self.pipeline = KPipeline(lang_code="i", repo_id=KOKORO_REPO_ID)
        self.preprocessor = preprocessor

    def synthesize(self, text: str, output_path: str, voice: str = DEFAULT_VOICE, speed: float = DEFAULT_SPEED) -> SynthesisResult:
        normalized_text = self.preprocessor.normalize(text, pronunciation_mode=self.pronunciation_mode)
        if normalized_text is None:
            raise RuntimeError("TTS preprocessing returned None")

        if isinstance(normalized_text, list):
            normalized_text = " ".join(str(part).strip() for part in normalized_text if str(part).strip())

        if not isinstance(normalized_text, str):
            normalized_text = str(normalized_text)

        normalized_text = normalized_text.strip()

        if not normalized_text:
            raise RuntimeError("No speakable text after preprocessing")

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        effective_speed = self.preprocessor.speaking_speed(normalized_text, speed)
        try:
            generator = self.pipeline(normalized_text, voice=voice, speed=effective_speed)
        except TypeError:
            generator = self.pipeline(normalized_text, voice=voice)

        audio_chunks: list[np.ndarray] = []
        for _, _, audio in generator:
            if audio is not None and len(audio) > 0:
                audio_chunks.append(np.asarray(audio, dtype=np.float32))

        if not audio_chunks:
            raise RuntimeError("No audio generated")

        full_audio = np.concatenate(audio_chunks)
        full_audio = normalize_audio(full_audio)
        pause_samples = int(SAMPLE_RATE * self.preprocessor.pause_ms(normalized_text) / 1000)
        if pause_samples > 0:
            full_audio = np.concatenate([full_audio, np.zeros(pause_samples, dtype=np.float32)])

        sf.write(output_path, full_audio, SAMPLE_RATE, subtype="PCM_16")
        return SynthesisResult(
            output_path=output_path,
            normalized_text=normalized_text,
            sample_rate=SAMPLE_RATE,
            engine=self.engine_name,
        )


class ChatterboxMultilingualEngine(BaseEngine):
    engine_name = "chatterbox_multilingual"
    pronunciation_mode: PronunciationMode = "plain"

    def __init__(self, preprocessor: TextPreprocessor, device: str) -> None:
        ChatterboxMultilingualTTS, _ = import_chatterbox_modules()
        self.model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        self.preprocessor = preprocessor
        self.device = device
        self.sample_rate = int(getattr(self.model, "sr", SAMPLE_RATE))

    @staticmethod
    def split_for_chatterbox(text: str, max_chars: int = 110) -> list[str]:
        if text is None:
            return []

        if not isinstance(text, str):
            text = str(text)

        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []

        clauses = re.split(r"(?<=[\.\!\?\;\:])\s+|(?<=,)\s+", text)
        parts: list[str] = []
        current = ""

        for clause in clauses:
            if clause is None:
                continue

            clause = str(clause).strip()
            if not clause:
                continue

            if len(clause) <= max_chars and not current:
                parts.append(clause)
                continue

            if len(current) + len(clause) + 1 <= max_chars:
                current = f"{current} {clause}".strip()
                continue

            if current:
                parts.append(current)
                current = ""

            if len(clause) <= max_chars:
                current = clause
                continue

            words = clause.split()
            chunk = ""
            for word in words:
                candidate = f"{chunk} {word}".strip()
                if len(candidate) <= max_chars:
                    chunk = candidate
                else:
                    if chunk:
                        parts.append(chunk)
                    chunk = word

            if chunk:
                current = chunk

        if current:
            parts.append(current)

        return [str(p).strip() for p in parts if isinstance(p, str) and p.strip()]

    def synthesize(self, text: str, output_path: str, voice: str = DEFAULT_VOICE, speed: float = DEFAULT_SPEED) -> SynthesisResult:
        normalized_text = self.preprocessor.normalize(text, pronunciation_mode=self.pronunciation_mode)
        if not normalized_text:
            raise RuntimeError("No speakable text after preprocessing")

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        generate_kwargs: dict[str, Any] = {
            "language_id": DEFAULT_LANGUAGE_ID,
            "cfg_weight": DEFAULT_CFG_WEIGHT,
            "exaggeration": DEFAULT_EXAGGERATION,
        }
        if DEFAULT_AUDIO_PROMPT_PATH:
            generate_kwargs["audio_prompt_path"] = DEFAULT_AUDIO_PROMPT_PATH



        parts = self.split_for_chatterbox(normalized_text, max_chars=CHATTERBOX_MAX_CHARS)
        if not parts:
            raise RuntimeError("No speakable text after chatterbox split")

        audio_chunks: list[np.ndarray] = []

        for idx, part in enumerate(parts):
            if not isinstance(part, str):
                part = str(part)

            part = part.strip()
            if not part:
                continue

            recovered_chunks = self._generate_part_with_retries(
                part=part,
                generate_kwargs=generate_kwargs,
                part_index=idx,
            )
            audio_chunks.extend(recovered_chunks)

            pause_ms = self.preprocessor.pause_ms(part)
            if idx < len(parts) - 1 and pause_ms > 0:
                pause_samples = int(self.sample_rate * pause_ms / 1000)
                if pause_samples > 0:
                    audio_chunks.append(np.zeros(pause_samples, dtype=np.float32))

        if not audio_chunks:
            raise RuntimeError("No audio generated by chatterbox")

        full_audio = np.concatenate(audio_chunks)
        pause_samples = int(self.sample_rate * self.preprocessor.pause_ms(normalized_text) / 1000)
        if pause_samples > 0:
            full_audio = np.concatenate([full_audio, np.zeros(pause_samples, dtype=np.float32)])

        sf.write(output_path, full_audio, self.sample_rate, subtype="PCM_16")
        return SynthesisResult(
            output_path=output_path,
            normalized_text=normalized_text,
            sample_rate=self.sample_rate,
            engine=self.engine_name,
        )

    def _generate_part_with_retries(
        self,
        part: str,
        generate_kwargs: dict[str, Any],
        part_index: int,
    ) -> list[np.ndarray]:
        try:
            return [self._generate_one_part(part, generate_kwargs)]
        except Exception as first_error:
            fallback_parts = self.split_for_chatterbox(
                stabilize_tts_text(part),
                max_chars=CHATTERBOX_RETRY_MAX_CHARS,
            )
            if len(fallback_parts) <= 1 and fallback_parts == [part]:
                raise RuntimeError(
                    f"Chatterbox failed on chunk {part_index + 1}: {type(first_error).__name__}: {first_error}"
                ) from first_error

            log_tts_event(
                "chatterbox_chunk_retry",
                part_index=part_index,
                original_chars=len(part),
                retry_chunks=len(fallback_parts),
                error=f"{type(first_error).__name__}: {first_error}",
            )

            recovered: list[np.ndarray] = []
            for retry_index, retry_part in enumerate(fallback_parts):
                retry_part = retry_part.strip()
                if not retry_part:
                    continue
                try:
                    recovered.append(self._generate_one_part(retry_part, generate_kwargs))
                except Exception as retry_error:
                    raise RuntimeError(
                        "Chatterbox failed on retry "
                        f"{part_index + 1}.{retry_index + 1}: "
                        f"{type(retry_error).__name__}: {retry_error}"
                    ) from retry_error

                pause_ms = min(self.preprocessor.pause_ms(retry_part), 90)
                if retry_index < len(fallback_parts) - 1 and pause_ms > 0:
                    recovered.append(np.zeros(int(self.sample_rate * pause_ms / 1000), dtype=np.float32))

            if not recovered:
                raise RuntimeError(
                    f"Chatterbox retry produced no audio for chunk {part_index + 1}"
                ) from first_error
            return recovered

    def _generate_one_part(self, part: str, generate_kwargs: dict[str, Any]) -> np.ndarray:
        stabilized = stabilize_tts_text(part)
        wav = self.model.generate(stabilized, **generate_kwargs)
        chunk = tensor_to_numpy(wav)
        chunk = normalize_audio(chunk)

        if chunk.size == 0:
            raise RuntimeError("Chatterbox returned empty audio")

        return chunk


class ChatterboxTurboEngine(BaseEngine):
    engine_name = "chatterbox_turbo"
    pronunciation_mode: PronunciationMode = "plain"

    def __init__(self, preprocessor: TextPreprocessor, device: str) -> None:
        _, ChatterboxTurboTTS = import_chatterbox_modules()
        self.model = ChatterboxTurboTTS.from_pretrained(device=device)
        self.preprocessor = preprocessor
        self.device = device
        self.sample_rate = int(getattr(self.model, "sr", SAMPLE_RATE))

    def synthesize(self, text: str, output_path: str, voice: str = DEFAULT_VOICE, speed: float = DEFAULT_SPEED) -> SynthesisResult:
        if not DEFAULT_AUDIO_PROMPT_PATH:
            raise RuntimeError(
                "Chatterbox Turbo requires ASTRA_TTS_AUDIO_PROMPT_PATH pointing to a short reference clip"
            )

        normalized_text = self.preprocessor.normalize(text, pronunciation_mode=self.pronunciation_mode)
        if not normalized_text:
            raise RuntimeError("No speakable text after preprocessing")

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        wav = self.model.generate(normalized_text, audio_prompt_path=DEFAULT_AUDIO_PROMPT_PATH)
        full_audio = tensor_to_numpy(wav)
        full_audio = normalize_audio(full_audio)
        pause_samples = int(self.sample_rate * self.preprocessor.pause_ms(normalized_text) / 1000)
        if pause_samples > 0:
            full_audio = np.concatenate([full_audio, np.zeros(pause_samples, dtype=np.float32)])

        sf.write(output_path, full_audio, self.sample_rate, subtype="PCM_16")
        return SynthesisResult(
            output_path=output_path,
            normalized_text=normalized_text,
            sample_rate=self.sample_rate,
            engine=self.engine_name,
        )


def stabilize_tts_text(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    text = re.sub(r"([!?]){2,}", r"\1", text)
    text = re.sub(r"\.{3,}", ".", text)
    text = text.strip(" \t\r\n")
    if text and text[-1] not in ".!?;:":
        text += "."
    return text


def log_tts_event(event: str, **fields: Any) -> None:
    payload = {"type": "tts_worker", "event": event, **fields}
    print(json.dumps(payload, ensure_ascii=False), file=sys.stderr, flush=True)


def tensor_to_numpy(value: Any) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu().float().numpy()
    array = np.asarray(value, dtype=np.float32)
    return np.squeeze(array)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.98:
        return audio / peak * 0.98
    return audio



class TtsEngine:
    def __init__(self) -> None:
        self.preprocessor = TextPreprocessor()
        self.device = select_device(DEFAULT_DEVICE)
        print(
            f"TTS device selection requested={DEFAULT_DEVICE} resolved={self.device} "
            f"torch_cuda_available={torch.cuda.is_available() if torch is not None else False}",
            file=sys.stderr,
            flush=True,
        )
        self.engine_name = resolve_engine_name(DEFAULT_ENGINE)
        self.engine = self._build_engine(self.engine_name)
        self._kokoro_fallback: KokoroEngine | None = None

    def _build_engine(self, engine_name: EngineName) -> BaseEngine:
        errors: list[str] = []
        if engine_name == "auto":
            for candidate in auto_engine_candidates():
                try:
                    return self._build_specific_engine(candidate)
                except Exception as exc:
                    errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
            raise RuntimeError("No TTS engine could be initialized. " + " | ".join(errors))
        return self._build_specific_engine(engine_name)

    def _build_specific_engine(self, engine_name: EngineName) -> BaseEngine:
        if engine_name == "kokoro":
            return KokoroEngine(self.preprocessor)
        if engine_name == "chatterbox_multilingual":
            return ChatterboxMultilingualEngine(self.preprocessor, self.device)
        if engine_name == "chatterbox_turbo":
            return ChatterboxTurboEngine(self.preprocessor, self.device)
        raise RuntimeError(f"Unsupported TTS engine: {engine_name}")

    def synthesize(self, text: str, output_path: str, voice: str = DEFAULT_VOICE, speed: float = DEFAULT_SPEED) -> SynthesisResult:
        try:
            return self.engine.synthesize(text=text, output_path=output_path, voice=voice, speed=speed)
        except Exception as primary_error:
            if self.engine.engine_name == "kokoro":
                raise

            log_tts_event(
                "primary_engine_failed_using_kokoro_fallback",
                engine=self.engine.engine_name,
                error=f"{type(primary_error).__name__}: {primary_error}",
            )
            fallback = self._get_kokoro_fallback()
            try:
                return fallback.synthesize(text=text, output_path=output_path, voice=voice, speed=speed)
            except Exception as fallback_error:
                raise RuntimeError(
                    "Primary TTS engine failed and Kokoro fallback also failed: "
                    f"primary={type(primary_error).__name__}: {primary_error}; "
                    f"fallback={type(fallback_error).__name__}: {fallback_error}"
                ) from fallback_error

    def _get_kokoro_fallback(self) -> KokoroEngine:
        if self._kokoro_fallback is None:
            self._kokoro_fallback = KokoroEngine(self.preprocessor)
        return self._kokoro_fallback


def auto_engine_candidates() -> list[EngineName]:
    candidates: list[EngineName] = []
    if DEFAULT_AUDIO_PROMPT_PATH:
        candidates.append("chatterbox_turbo")
    candidates.append("chatterbox_multilingual")
    candidates.append("kokoro")
    return candidates


def resolve_engine_name(value: str) -> EngineName:
    normalized = (value or "auto").strip().lower()
    if normalized in {"auto", "kokoro", "chatterbox_multilingual", "chatterbox_turbo"}:
        return normalized  # type: ignore[return-value]
    raise RuntimeError(
        "ASTRA_TTS_ENGINE must be one of: auto, kokoro, chatterbox_multilingual, chatterbox_turbo"
    )


def run_server() -> None:
    engine = TtsEngine()
    print(
        f"TTS worker ready engine={engine.engine.engine_name} device={engine.device}",
        file=sys.stderr,
        flush=True,
    )

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        request: dict[str, Any] = {}
        try:
            request = json.loads(line)
            response = handle_request(engine, request)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            response = {
                "ok": False,
                "request_id": safe_get(locals().get("request"), "request_id"),
                "segment_id": safe_get(locals().get("request"), "segment_id"),
                "sequence": safe_get(locals().get("request"), "sequence"),
                "error": f"{type(exc).__name__}: {exc}",
            }

        print(json.dumps(response, ensure_ascii=False), flush=True)


def handle_request(engine: TtsEngine, request: dict[str, Any]) -> dict[str, Any]:
    request_id = str(request["request_id"])
    segment_id = str(request["segment_id"])
    sequence = int(request.get("sequence", 0))
    text = str(request["text"])
    output_path = str(request["output_path"])
    voice = str(request.get("voice", DEFAULT_VOICE))
    speed = float(request.get("speed", DEFAULT_SPEED))

    result = engine.synthesize(text=text, output_path=output_path, voice=voice, speed=speed)

    return {
        "ok": True,
        "request_id": request_id,
        "segment_id": segment_id,
        "sequence": sequence,
        "output_path": result.output_path,
        "normalized_text": result.normalized_text,
        "sample_rate": result.sample_rate,
        "engine": result.engine,
    }


def safe_get(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key, "")
    return ""


def run_cli() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: tts_worker.py <text> <output_path> OR tts_worker.py --server")

    engine = TtsEngine()
    result = engine.synthesize(text=sys.argv[1], output_path=sys.argv[2])
    print(result.output_path)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--server":
        run_server()
    else:
        run_cli()
