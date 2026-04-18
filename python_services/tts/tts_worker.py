from __future__ import annotations

import json
import logging
import os
import re
import sys
import traceback
import warnings
from dataclasses import dataclass
from typing import Any

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

from kokoro import KPipeline
import numpy as np
import soundfile as sf


SAMPLE_RATE = 24000
KOKORO_REPO_ID = os.environ.get("ASTRA_TTS_REPO_ID", "hexgrad/Kokoro-82M")
DEFAULT_VOICE = os.environ.get("ASTRA_TTS_VOICE", "if_sara")
DEFAULT_SPEED = float(os.environ.get("ASTRA_TTS_SPEED", "0.96"))


@dataclass(frozen=True)
class VoiceProfile:
    acknowledgement_pause_ms: int = 90
    short_reply_pause_ms: int = 110
    sentence_pause_ms: int = 140
    long_sentence_pause_ms: int = 185
    comma_pause_ms: int = 95
    semicolon_pause_ms: int = 165
    colon_pause_ms: int = 185
    question_pause_ms: int = 240
    exclamation_pause_ms: int = 215
    list_item_pause_ms: int = 210
    short_ack_speed: float = 0.93
    short_reply_speed: float = 0.95
    long_explanation_speed: float = 0.98


@dataclass(frozen=True)
class SynthesisResult:
    output_path: str
    normalized_text: str
    sample_rate: int


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
        r"^(certo|ok|okay|va bene|perfetto|capito|chiaro|bene|sì|si|assolutamente)[.!]?$",
        re.IGNORECASE,
    )
    _conversational_prefix_pattern = re.compile(
        r"^(certo|ok|okay|va bene|perfetto|capito|chiaro|bene|sì|si),\s+",
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

    def __init__(self, profile: VoiceProfile | None = None) -> None:
        self.profile = profile or VoiceProfile()
        self.number_normalizer = ItalianNumberNormalizer()

    def normalize(self, text: str) -> str:
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
        if self._is_short_reply(stripped):
            return min(requested_speed, self.profile.short_reply_speed)
        if len(stripped) > 220:
            return max(requested_speed, self.profile.long_explanation_speed)
        return requested_speed

    def _normalize_punctuation(self, text: str) -> str:
        text = text.replace("\u2026", "...")
        text = re.sub(r"\.{4,}", "...", text)
        text = re.sub(r"([!?]){2,}", r"\1", text)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = re.sub(r"([,.;:!?])([^\s,.;:!?])", r"\1 \2", text)
        text = re.sub(r"\s+-\s+", ", ", text)
        text = re.sub(r"\s*/\s*", " o ", text)
        return text

    def _shape_conversational_prefixes(self, text: str) -> str:
        return self._conversational_prefix_pattern.sub(
            lambda match: f"{match.group(1).capitalize()}, ",
            text,
        )

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


class TtsEngine:
    def __init__(self) -> None:
        self.pipeline = KPipeline(lang_code="i", repo_id=KOKORO_REPO_ID)
        self.preprocessor = TextPreprocessor()

    def synthesize(
        self,
        text: str,
        output_path: str,
        voice: str = DEFAULT_VOICE,
        speed: float = DEFAULT_SPEED,
    ) -> SynthesisResult:
        normalized_text = self.preprocessor.normalize(text)
        if not normalized_text:
            raise RuntimeError("No speakable text after preprocessing")

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        try:
            effective_speed = self.preprocessor.speaking_speed(normalized_text, speed)
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
        full_audio = self._normalize_audio(full_audio)

        pause_samples = int(SAMPLE_RATE * self.preprocessor.pause_ms(normalized_text) / 1000)
        if pause_samples > 0:
            full_audio = np.concatenate([full_audio, np.zeros(pause_samples, dtype=np.float32)])

        sf.write(output_path, full_audio, SAMPLE_RATE, subtype="PCM_16")

        return SynthesisResult(
            output_path=output_path,
            normalized_text=normalized_text,
            sample_rate=SAMPLE_RATE,
        )

    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 0.98:
            return audio / peak * 0.98
        return audio


def run_server() -> None:
    engine = TtsEngine()
    print("TTS worker ready", file=sys.stderr, flush=True)

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
