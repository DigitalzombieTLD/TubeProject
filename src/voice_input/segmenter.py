"""Energy-based utterance segmentation driven by block counts (no wall clock)."""
from collections import deque
from typing import Optional

from .audio import RATE, SAMPLE_WIDTH


class Segmenter:
    """Feed fixed-size mono blocks; returns a finished utterance (bytes) or None."""

    def __init__(self, block_seconds: float, threshold: float, silence: float,
                 preroll: float, min_utterance: float, max_utterance: float = 60.0):
        self.block_seconds = block_seconds
        self.threshold = threshold
        self.silence_blocks = max(1, round(silence / block_seconds))
        self.min_blocks = max(1, round(min_utterance / block_seconds))
        self.max_blocks = max(1, round(max_utterance / block_seconds))
        self._preroll = deque(maxlen=max(0, round(preroll / block_seconds)))
        self._blocks: list = []
        self._speaking = False
        self._silent_run = 0
        self._voiced_blocks = 0

    @property
    def speaking(self) -> bool:
        return self._speaking

    def feed(self, block: bytes, block_rms: float) -> Optional[bytes]:
        voiced = block_rms >= self.threshold
        if not self._speaking:
            if not voiced:
                self._preroll.append(block)
                return None
            self._blocks = list(self._preroll) + [block]
            self._preroll.clear()
            self._speaking = True
            self._silent_run = 0
            self._voiced_blocks = 1
            return None

        self._blocks.append(block)
        if voiced:
            self._silent_run = 0
            self._voiced_blocks += 1
        else:
            self._silent_run += 1
        if self._silent_run >= self.silence_blocks or len(self._blocks) >= self.max_blocks:
            return self._finish()
        return None

    def flush(self) -> Optional[bytes]:
        """Finish any in-progress utterance (e.g. at shutdown)."""
        return self._finish() if self._speaking else None

    def _finish(self) -> Optional[bytes]:
        blocks = self._blocks
        if self._silent_run:
            blocks = blocks[: len(blocks) - self._silent_run]
        voiced = self._voiced_blocks
        self._blocks, self._speaking, self._silent_run, self._voiced_blocks = [], False, 0, 0
        self._preroll.clear()
        if voiced < self.min_blocks:
            return None
        return b"".join(blocks)


def duration_seconds(mono: bytes, rate: int = RATE) -> float:
    return len(mono) / (SAMPLE_WIDTH * rate)
