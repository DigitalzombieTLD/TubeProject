"""Long-lived OpenAI Whisper worker. Run it with the interpreter of the *Whisper* environment.

Speaks the same JSON Lines protocol as ``sensevoice_worker.py`` (ready/error events, one reply per
request, ``{"cmd": "shutdown"}``), so the app drives both back ends with the same session code.
``whisper.load_model()`` runs exactly once at startup; every utterance then reuses the loaded model.

This is the official ``openai-whisper`` package running locally on PyTorch (CPU by default). It does not
use the AX650N cards and does not call any OpenAI web service. If the model weights are not in the cache,
``whisper.load_model()`` downloads them once, here, at startup (default cache: ~/.cache/whisper).
"""
import argparse
import json
import os
import sys

try:
    from sensevoice_worker import SAMPLE_RATE, STARTUP_FAILURE, serve  # run as a script
except ImportError:  # imported as part of the package
    from .sensevoice_worker import SAMPLE_RATE, STARTUP_FAILURE, serve


class WhisperEngine:
    """Adapter over the Whisper Python API. ``whisper_module``/``np`` can be injected for tests."""

    def __init__(self, model: str, task: str = "transcribe", device: str = "cpu",
                 download_root: str = "", whisper_module=None, np=None):
        if whisper_module is None:
            import whisper as whisper_module
        if np is None:
            import numpy as np
        self._np, self._task = np, task
        kwargs = {"device": device}
        if download_root:
            kwargs["download_root"] = download_root
        self._model = whisper_module.load_model(model, **kwargs)  # the only model load
        self._fp16 = device != "cpu"  # fp16 is unsupported on CPU; Whisper would warn and fall back
        tokenizer = getattr(whisper_module, "tokenizer", None)
        codes = getattr(tokenizer, "LANGUAGES", None)
        self.languages = ["auto"] + list(codes) if codes else None

    def transcribe(self, pcm: bytes, language: str) -> str:
        np = self._np
        # int16 -> float32 in [-1, 1) at 16 kHz: the array form Whisper's transcribe() accepts directly,
        # so no file or ffmpeg round trip is needed for the live PCM path.
        wave = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        result = self._model.transcribe(
            wave, language=None if language == "auto" else language, task=self._task,
            fp16=self._fp16, condition_on_previous_text=False)
        return result.get("text", "")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Persistent Whisper worker (internal)")
    p.add_argument("--model", required=True)
    p.add_argument("--task", default="transcribe")
    p.add_argument("--device", default="cpu")
    p.add_argument("--download-root", default="")
    args = p.parse_args(argv)

    out = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    try:
        engine = WhisperEngine(args.model, args.task, args.device, args.download_root)
    except BaseException as exc:
        out.write(json.dumps({"event": "error", "error": f"{type(exc).__name__}: {exc}"}) + "\n")
        out.flush()
        return STARTUP_FAILURE
    return serve(engine, sys.stdin, out)


if __name__ == "__main__":
    raise SystemExit(main())
