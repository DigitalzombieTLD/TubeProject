"""arecord-based microphone capture with stderr forwarding."""
import subprocess
import sys
import threading
from typing import Iterator, List

from .audio import CHANNELS, RATE


def build_arecord_command(device: str, rate: int = RATE, channels: int = CHANNELS) -> List[str]:
    return ["arecord", "-D", device, "-t", "raw", "-f", "S16_LE",
            "-r", str(rate), "-c", str(channels)]


class Capture:
    def __init__(self, device: str, block_ms: int = 20):
        self.cmd = build_arecord_command(device)
        self.block_bytes = RATE * block_ms // 1000 * CHANNELS * 2
        self._proc = None

    def start(self) -> None:
        try:
            self._proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE, bufsize=0)
        except FileNotFoundError as exc:
            raise RuntimeError("arecord not found; install alsa-utils on the host") from exc
        threading.Thread(target=self._forward_stderr, name="arecord-stderr", daemon=True).start()

    def _forward_stderr(self) -> None:
        for line in iter(self._proc.stderr.readline, b""):
            print(f"[arecord] {line.decode(errors='replace').rstrip()}", file=sys.stderr, flush=True)

    def blocks(self) -> Iterator[bytes]:
        out = self._proc.stdout
        while True:
            buf = b""
            while len(buf) < self.block_bytes:
                chunk = out.read(self.block_bytes - len(buf))
                if not chunk:
                    raise RuntimeError(
                        f"arecord stopped (exit {self._proc.poll()}); check the ALSA device "
                        "with 'arecord -L' and the [arecord] messages above")
                buf += chunk
            yield buf

    def stop(self) -> None:
        p = self._proc
        if p and p.poll() is None:
            p.terminate()
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
