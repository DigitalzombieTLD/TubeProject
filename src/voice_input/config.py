"""Settings, JSON config loading."""
from dataclasses import dataclass, fields
import json
from pathlib import Path
import re

LANGUAGES = ("auto", "zh", "en", "yue", "ja", "ko")
ENGINES = ("persistent", "cli", "whisper")
WHISPER_MODELS = ("tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en",
                  "large", "large-v1", "large-v2", "large-v3", "turbo")
WHISPER_TASKS = ("transcribe", "translate")
WHISPER_DEVICES = ("cpu", "cuda")
OUTPUTS = ("text", "jsonl")


@dataclass
class Settings:
    device: str = "plughw:CARD=Array,DEV=0"
    channel: int = 0
    silence: float = 0.8
    threshold: float = 500.0
    preroll: float = 0.2
    min_utterance: float = 0.3
    max_utterance: float = 30.0
    queue_size: int = 4
    queue_policy: str = "drop-newest"
    language: str = "en"
    chip: str = "ax650"
    sensevoice_python: str = ""  # empty: use the interpreter running this program
    sensevoice_main: str = ""    # path to SenseVoice python/main.py (required)
    timeout: float = 120.0
    startup_timeout: float = 120.0
    engine: str = "persistent"   # "cli" = legacy fallback, one main.py process (model reload) per utterance
    output: str = "text"         # "jsonl" = one JSON object per transcript on stdout
    sensevoice_model_dir: str = ""  # empty: <SenseVoice python dir>/models/SenseVoice/sensevoice_<chip>
    whisper_python: str = ""     # empty: use the interpreter running this program (needs openai-whisper)
    whisper_model: str = ""      # empty: base.en for English, base otherwise; or a checkpoint path
    whisper_language: str = ""   # empty: use `language`; otherwise any Whisper code (e.g. de) or "auto"
    whisper_task: str = "transcribe"  # "translate" = to English (multilingual models only)
    whisper_device: str = "cpu"  # official Whisper runs on PyTorch; cpu is the only verified-by-design choice
    whisper_download_root: str = ""  # empty: Whisper default, ~/.cache/whisper (or $XDG_CACHE_HOME/whisper)

    def effective_whisper_language(self) -> str:
        return self.whisper_language or self.language

    def effective_whisper_model(self) -> str:
        if self.whisper_model:
            return self.whisper_model
        return "base.en" if self.effective_whisper_language() == "en" else "base"

    def validate(self) -> None:
        if self.channel not in (0, 1):
            raise ValueError("channel must be 0 or 1")
        if self.engine == "whisper":
            self._validate_whisper()
        elif self.language not in LANGUAGES:
            raise ValueError(f"language must be one of {LANGUAGES}")
        if self.engine not in ENGINES:
            raise ValueError(f"engine must be one of {ENGINES}")
        if self.output not in OUTPUTS:
            raise ValueError(f"output must be one of {OUTPUTS}")
        if self.queue_size < 1:
            raise ValueError("queue_size must be >= 1")
        for name in ("silence", "threshold", "preroll", "min_utterance", "max_utterance", "timeout",
                     "startup_timeout"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.engine != "whisper" and not self.sensevoice_main:
            raise ValueError("sensevoice_main is required (--sensevoice-main or config file)")

    def _validate_whisper(self) -> None:
        lang = self.effective_whisper_language()
        if lang != "auto" and not re.fullmatch(r"[a-z]{2,3}", lang):
            raise ValueError("whisper language must be 'auto' or a Whisper language code such as en, de, ja")
        if self.whisper_task not in WHISPER_TASKS:
            raise ValueError(f"whisper_task must be one of {WHISPER_TASKS}")
        if self.whisper_device not in WHISPER_DEVICES:
            raise ValueError(f"whisper_device must be one of {WHISPER_DEVICES}")
        model = self.effective_whisper_model()
        is_path = "/" in model or model.endswith(".pt")
        if not is_path and model not in WHISPER_MODELS:
            raise ValueError(f"whisper_model must be one of {WHISPER_MODELS} or a path to a checkpoint")
        if model.endswith(".en") and not is_path:
            if lang not in ("en", "auto"):
                raise ValueError(f"English-only model {model!r} cannot be used with language {lang!r}; "
                                 "choose a multilingual model (e.g. base, small)")
            if self.whisper_task == "translate":
                raise ValueError(f"English-only model {model!r} cannot translate; use a multilingual model")


def load_config(path) -> dict:
    data = json.loads(Path(path).expanduser().read_text())
    known = {f.name for f in fields(Settings)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    return data
