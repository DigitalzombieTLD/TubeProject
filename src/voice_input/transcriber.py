"""SenseVoice back ends.

* ``SenseVoiceSession`` (default): one long-lived worker process that loads the model once.
* ``SenseVoiceCliTranscriber`` (fallback): launches SenseVoice ``main.py`` per utterance, so the
  runtime and model are reloaded every time. Kept only as an explicit ``--engine cli`` fallback.
"""
import base64
from dataclasses import dataclass
import json
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time
from typing import List, Optional

from .audio import RATE, SAMPLE_WIDTH, write_wav

RESULT_PREFIX = "ASR result:"


class TranscriptionError(RuntimeError):
    pass


@dataclass
class Transcription:
    text: str
    inference_seconds: Optional[float] = None  # model-reported; None for the CLI fallback
    wall_seconds: float = 0.0                  # time spent inside transcribe()


def build_command(python: str, main_py: Path, wav: Path, chip: str, language: str) -> List[str]:
    return [python, str(main_py), "-i", str(wav), "--chip", chip, "-l", language]


def parse_result(output: str):
    lines = [l for l in output.splitlines() if l.startswith(RESULT_PREFIX)]
    return lines[-1][len(RESULT_PREFIX):].strip() if lines else None


class SenseVoiceCliTranscriber:
    def __init__(self, python: str, main_py: Path, chip: str = "ax650",
                 language: str = "en", timeout: float = 120.0):
        self.python, self.main_py = python, Path(main_py)
        self.chip, self.language, self.timeout = chip, language, timeout
        self._proc = None
        self._lock = threading.Lock()
        self._closed = False

    def transcribe(self, mono: bytes) -> Transcription:
        began = time.monotonic()
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
            return Transcription(text, None, time.monotonic() - began)

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


WORKER_SCRIPT = Path(__file__).with_name("sensevoice_worker.py")


class SenseVoiceSession:
    """Owns one persistent SenseVoice worker process (see sensevoice_worker.py for the protocol).

    start() loads the model once; transcribe() is then called for every utterance without any
    re-initialisation. A worker that dies or hangs is killed and transparently restarted on the next
    call. transcribe() is serialised, so utterances are processed strictly in order.
    """

    label = "SenseVoice"

    def __init__(self, python: str, sensevoice_dir: Path, chip: str = "ax650", language: str = "en",
                 timeout: float = 120.0, startup_timeout: float = 120.0, model_dir: str = "",
                 worker_script: Path = WORKER_SCRIPT, extra_args: Optional[List[str]] = None):
        self.python, self.sensevoice_dir = python, Path(sensevoice_dir)
        self.chip, self.language = chip, language
        self.timeout, self.startup_timeout = timeout, startup_timeout
        self.model_dir = model_dir
        self.worker_script = Path(worker_script)
        self.extra_args = list(extra_args or [])
        self.starts = 0  # number of worker launches (1 for the whole session unless it had to recover)
        self._proc = None
        self._lines: "queue.Queue[Optional[str]]" = queue.Queue()
        self._reader = None
        self._lock = threading.Lock()   # serialises transcribe()/start()
        self._state = threading.Lock()  # guards _proc/_closed for shutdown()
        self._closed = False
        self._next_id = 0

    def command(self) -> List[str]:
        cmd = [self.python, str(self.worker_script), "--sensevoice-dir", str(self.sensevoice_dir),
               "--chip", self.chip]
        if self.model_dir:
            cmd += ["--model-dir", self.model_dir]
        return cmd + self.extra_args

    def start(self) -> None:
        """Launch the worker and wait until the model is loaded. Raises TranscriptionError."""
        with self._lock:
            self._start_locked()

    def _start_locked(self) -> None:
        with self._state:
            if self._closed:
                raise TranscriptionError("transcriber is shut down")
            if self._proc is not None and self._proc.poll() is None:
                return
        cmd = self.command()
        lines: "queue.Queue[Optional[str]]" = queue.Queue()
        try:
            # Own session: a terminal Ctrl-C reaches only this program, which then shuts the worker
            # down itself. If this program dies, the worker sees stdin EOF and exits.
            proc = subprocess.Popen(cmd, cwd=self.sensevoice_dir, text=True, encoding="utf-8",
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    start_new_session=True)
        except OSError as exc:
            raise TranscriptionError(f"cannot launch {cmd[0]}: {exc}") from exc
        reader = threading.Thread(target=self._read, args=(proc, lines),
                                  name="sensevoice-reader", daemon=True)
        with self._state:
            if self._closed:
                self._reap(proc, None)
                raise TranscriptionError("transcriber is shut down")
            self._proc, self._lines, self._reader = proc, lines, reader
        self.starts += 1
        reader.start()
        try:
            msg = self._await(lambda m: "event" in m, self.startup_timeout, "startup")
        except TranscriptionError:
            self._discard()
            raise
        if msg["event"] != "ready":
            self._discard()
            raise TranscriptionError(f"{self.label} failed to start: {msg.get('error', msg)}")

    @staticmethod
    def _read(proc, lines) -> None:
        try:
            for line in proc.stdout:
                lines.put(line)
        except (OSError, ValueError):
            pass
        finally:
            lines.put(None)  # EOF: worker exited or was killed

    def _await(self, match, timeout: float, what: str) -> dict:
        deadline = time.monotonic() + timeout
        lines, proc = self._lines, self._proc
        while True:
            try:
                line = lines.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                raise TranscriptionError(f"{self.label} {what} timed out after {timeout}s") from None
            if line is None:
                lines.put(None)
                code = proc.poll() if proc else None
                if self._closed:
                    raise TranscriptionError("transcriber is shut down")
                raise TranscriptionError(f"{self.label} worker exited unexpectedly (code {code}) during {what}")
            try:
                msg = json.loads(line)
            except ValueError:
                continue  # not protocol traffic
            if isinstance(msg, dict) and match(msg):
                return msg

    def transcribe(self, mono: bytes) -> Transcription:
        began = time.monotonic()
        with self._lock:
            if len(mono) < SAMPLE_WIDTH:
                raise TranscriptionError("empty audio")
            self._start_locked()  # no-op while the worker is alive; restarts after a crash/timeout
            self._next_id += 1
            req_id = self._next_id
            request = json.dumps({"id": req_id, "language": self.language,
                                  "pcm": base64.b64encode(mono).decode("ascii")}) + "\n"
            proc = self._proc
            if proc is None:
                raise TranscriptionError("transcriber is shut down")
            try:
                proc.stdin.write(request)
                proc.stdin.flush()
                msg = self._await(lambda m: m.get("id") == req_id, self.timeout, "inference")
            except (OSError, ValueError) as exc:
                self._discard()
                raise TranscriptionError(f"{self.label} worker pipe failed: {exc}") from exc
            except TranscriptionError:
                self._discard()  # hung or dead worker: kill it; the next utterance restarts it
                raise
            if not msg.get("ok"):
                raise TranscriptionError(f"{self.label} inference failed: {msg.get('error')}")
            return Transcription(msg.get("text", ""), msg.get("inference_seconds"),
                                 time.monotonic() - began)

    def _discard(self) -> None:
        with self._state:
            proc, reader = self._proc, self._reader
            self._proc = self._reader = None
        self._reap(proc, reader)

    @staticmethod
    def _reap(proc, reader, grace: float = 0.0) -> None:
        """Stop a worker process, then join its reader thread and release the pipe."""
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
        except OSError:
            pass
        if grace:
            try:
                proc.wait(grace)
            except subprocess.TimeoutExpired:
                pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if reader is not None and reader.is_alive():
            reader.join(5)
        if proc.stdout:
            proc.stdout.close()

    def shutdown(self) -> None:
        """Refuse new work, wake any waiting transcribe(), and stop the worker. Idempotent."""
        with self._state:
            self._closed = True
            proc, reader = self._proc, self._reader
            self._proc = self._reader = None
            lines = self._lines
        lines.put(None)  # unblock a transcribe() currently waiting
        self._reap(proc, reader, grace=2.0)


WHISPER_WORKER_SCRIPT = Path(__file__).with_name("whisper_worker.py")


class WhisperSession(SenseVoiceSession):
    """Same persistent-worker protocol, backed by whisper_worker.py (official OpenAI Whisper, PyTorch)."""

    label = "Whisper"

    def __init__(self, python: str, model: str, language: str = "en", task: str = "transcribe",
                 device: str = "cpu", download_root: str = "", timeout: float = 120.0,
                 startup_timeout: float = 120.0, worker_script: Path = WHISPER_WORKER_SCRIPT,
                 extra_args: Optional[List[str]] = None):
        super().__init__(python, Path.cwd(), language=language, timeout=timeout,
                         startup_timeout=startup_timeout, worker_script=worker_script,
                         extra_args=extra_args)
        self.model, self.task, self.device, self.download_root = model, task, device, download_root

    def command(self) -> List[str]:
        cmd = [self.python, str(self.worker_script), "--model", self.model,
               "--task", self.task, "--device", self.device]
        if self.download_root:
            cmd += ["--download-root", self.download_root]
        return cmd + self.extra_args
