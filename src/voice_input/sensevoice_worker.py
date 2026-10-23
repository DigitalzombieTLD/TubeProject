"""Long-lived SenseVoice worker. Run it with the SenseVoice interpreter, not imported by the app.

It loads ``SenseVoiceAx`` (from AXERA's SenseVoice ``python/`` directory, the same class ``main.py``
uses) exactly once, then serves JSON Lines requests on stdin until stdin closes:

    request : {"id": 1, "language": "en", "pcm": "<base64 mono S16_LE 16 kHz>"}
              {"cmd": "shutdown"}
    response: {"event": "ready"}                                  (once, after model load)
              {"event": "error", "error": "..."}                  (startup failure, then exit 3)
              {"id": 1, "ok": true, "text": "...", "inference_seconds": 0.13, "audio_seconds": 7.2}
              {"id": 1, "ok": false, "error": "..."}              (worker stays alive)

Protocol messages go to a private duplicate of the original stdout; fd 1 is redirected to stderr so
library banners can never corrupt the stream. This file uses only the standard library at import
time (numpy / SenseVoiceAx are imported inside the SenseVoice environment when loading the model).
"""
import argparse
import base64
import json
import os
from pathlib import Path
import sys
import time

SAMPLE_RATE = 16000
STARTUP_FAILURE = 3
MODEL_FILES = ("sensevoice.axmodel", "am.mvn", "tokens.txt", "chn_jpn_yue_eng_ko_spectok.bpe.model")


def default_model_root(sensevoice_dir: Path, chip: str) -> Path:
    """Where SenseVoice's download_utils.download_model('SenseVoice') + main.py put the model files.

    Resolved here instead of calling download_model so this worker can never start a download.
    """
    return Path(sensevoice_dir) / "models" / "SenseVoice" / f"sensevoice_{chip}"


def check_model_files(model_root: Path) -> None:
    missing = [n for n in MODEL_FILES if not (Path(model_root) / n).is_file()]
    if missing:
        raise FileNotFoundError(f"SenseVoice model files missing in {model_root}: {', '.join(missing)}")


class AxEngine:
    """Thin adapter over SenseVoiceAx; model/runtime initialisation happens once in __init__."""

    def __init__(self, sensevoice_dir: Path, model_root: Path):
        sys.path.insert(0, str(sensevoice_dir))
        import numpy as np
        from SenseVoiceAx import SenseVoiceAx  # same constructor arguments as SenseVoice main.py
        self._np = np
        self._model = SenseVoiceAx(
            str(model_root / "sensevoice.axmodel"), str(model_root / "am.mvn"),
            str(model_root / "tokens.txt"), str(model_root / "chn_jpn_yue_eng_ko_spectok.bpe.model"),
            max_seq_len=256, beam_size=3, hot_words=None, streaming=False)
        self.languages = list(self._model.language_options)

    def transcribe(self, pcm: bytes, language: str) -> str:
        np = self._np
        # int16 -> float32 in [-1, 1): what librosa.load gives main.py for 16-bit 16 kHz audio.
        wave = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        # infer() builds fresh fbank state per call; the non-streaming path keeps no state between calls.
        return self._model.infer((wave, SAMPLE_RATE), language)


def serve(engine, stdin, out) -> int:
    def send(msg):
        out.write(json.dumps(msg, ensure_ascii=False) + "\n")
        out.flush()

    send({"event": "ready"})
    for line in stdin:
        if not line.strip():
            continue
        req_id = None
        try:
            req = json.loads(line)
            if req.get("cmd") == "shutdown":
                break
            req_id = req.get("id")
            pcm = base64.b64decode(req["pcm"], validate=True)
            if len(pcm) < 2:
                raise ValueError("empty audio")
            language = req.get("language", "auto")
            if language not in engine.languages:
                raise ValueError(f"unsupported language {language!r}")
            start = time.monotonic()
            text = engine.transcribe(pcm[: len(pcm) - len(pcm) % 2], language)
            send({"id": req_id, "ok": True, "text": text.strip(),
                  "inference_seconds": time.monotonic() - start,
                  "audio_seconds": len(pcm) // 2 / SAMPLE_RATE})
        except Exception as exc:  # recoverable: report and keep serving
            send({"id": req_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Persistent SenseVoice worker (internal)")
    p.add_argument("--sensevoice-dir", required=True, help="directory containing SenseVoiceAx.py")
    p.add_argument("--chip", default="ax650")
    p.add_argument("--model-dir", default="", help="override the sensevoice_<chip> model directory")
    args = p.parse_args(argv)

    out = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    try:
        sv_dir = Path(args.sensevoice_dir)
        model_root = Path(args.model_dir) if args.model_dir else default_model_root(sv_dir, args.chip)
        check_model_files(model_root)
        engine = AxEngine(sv_dir, model_root)
    except BaseException as exc:
        out.write(json.dumps({"event": "error", "error": f"{type(exc).__name__}: {exc}"}) + "\n")
        out.flush()
        return STARTUP_FAILURE
    return serve(engine, sys.stdin, out)


if __name__ == "__main__":
    raise SystemExit(main())
