"""Pure audio helpers: channel extraction, RMS and WAV writing."""
from array import array
import math
import sys
import wave

RATE = 16000
CHANNELS = 2
SAMPLE_WIDTH = 2


def extract_channel(raw: bytes, channel: int, channels: int = CHANNELS) -> bytes:
    """Return the selected channel of interleaved S16_LE PCM as mono S16_LE bytes."""
    if not 0 <= channel < channels:
        raise ValueError(f"channel must be in 0..{channels - 1}, got {channel}")
    frame_bytes = SAMPLE_WIDTH * channels
    raw = raw[: len(raw) - len(raw) % frame_bytes]
    samples = array("h")
    samples.frombytes(raw)
    if sys.byteorder != "little":
        samples.byteswap()
    mono = samples[channel::channels]
    if sys.byteorder != "little":
        mono.byteswap()
    return mono.tobytes()


def rms(mono: bytes) -> float:
    """RMS of mono S16_LE PCM; 0.0 when empty."""
    samples = array("h")
    samples.frombytes(mono[: len(mono) - len(mono) % SAMPLE_WIDTH])
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


def write_wav(path, mono: bytes, rate: int = RATE) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(rate)
        wav.writeframes(mono)
