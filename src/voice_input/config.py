"""Settings, JSON config loading."""
from dataclasses import dataclass, fields
import json
from pathlib import Path

LANGUAGES = ("auto", "zh", "en", "yue", "ja", "ko")
ENGINES = ("persistent", "cli")
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

    def validate(self) -> None:
        if self.channel not in (0, 1):
            raise ValueError("channel must be 0 or 1")
        if self.language not in LANGUAGES:
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
        if not self.sensevoice_main:
            raise ValueError("sensevoice_main is required (--sensevoice-main or config file)")


def load_config(path) -> dict:
    data = json.loads(Path(path).expanduser().read_text())
    known = {f.name for f in fields(Settings)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    return data
