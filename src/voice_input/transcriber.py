"""Runs the existing SenseVoice Python CLI on a WAV file."""
from pathlib import Path
import subprocess
import tempfile
import threading
from typing import List

from .audio import write_wav

RESULT_PREFIX = "ASR result:"


class TranscriptionError(RuntimeError):
    pass


def build_command(python: str, main_py: Path, wav: Path, chip: str, language: str) -> List[str]:
    return [python, str(main_py), "-i", str(wav), "--chip", chip, "-l", language]


def parse_result(output: str):
    lines = [l for l in output.splitlines() if l.startswith(RESULT_PREFIX)]
    return lines[-1][len(RESULT_PREFIX):].strip() if lines else None


class SenseVoiceTranscriber:
    def __init__(self, python: str, main_py: Path, chip: str = "ax650",
                 language: str = "en", timeout: float = 120.0):
        self.python, self.main_py = python, Path(main_py)
        self.chip, self.language, self.timeout = chip, language, timeout
        self._proc = None
        self._lock = threading.Lock()
        self._closed = False

    def transcribe(self, mono: bytes) -> str:
        with tempfile.TemporaryDirectory(prefix="tube-voice-") as tmp:
            wav = Path(tmp) / "utterance.wav"
            write_wav(wav, mono)
            cmd = build_command(self.python, self.main_py, wav, self.chip, self.language)
            with self._lock:
                if self._closed:
                    raise TranscriptionError("transcriber is shut down")
                try:
                    # SenseVoice main.py loads model paths relative to its own directory.
                    self._proc = subprocess.Popen(
                        cmd, cwd=self.main_py.parent, text=True,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                except OSError as exc:
                    raise TranscriptionError(f"cannot launch {cmd[0]}: {exc}") from exc
                proc = self._proc
            try:
                out, _ = proc.communicate(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                raise TranscriptionError(f"SenseVoice timed out after {self.timeout}s")
            finally:
                self._proc = None
            if proc.returncode != 0:
                raise TranscriptionError(
                    f"SenseVoice exited with {proc.returncode}:\n{(out or '').strip()}")
            text = parse_result(out or "")
            if text is None:
                raise TranscriptionError(f"no '{RESULT_PREFIX}' line in output:\n{(out or '').strip()}")
            return text

    def shutdown(self) -> None:
        """Stop any running SenseVoice subprocess and refuse new work."""
        with self._lock:
            self._closed = True
            proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(5)
            except subprocess.TimeoutExpired:
                proc.kill()
