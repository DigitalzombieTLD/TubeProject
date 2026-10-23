"""Transcript records and the stable text / JSON Lines output boundary."""
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from typing import Optional


@dataclass
class Utterance:
    """One queued job. ``seq`` is assigned when the utterance is segmented, so a gap in the numbers
    seen by a consumer means an utterance was dropped by the bounded queue."""
    seq: int
    pcm: bytes
    ended_at: float  # time.time() when the segmenter closed the utterance


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def transcript_record(utt: Utterance, text: str, language: str, audio_seconds: float,
                      inference_seconds: Optional[float], finished_at: float) -> dict:
    """``started_at`` is estimated as ended_at - audio_seconds (audio includes pre-roll and the
    trailing silence). ``latency_seconds`` is utterance end -> transcript ready (queue wait + inference
    + worker round trip)."""
    return {
        "type": "transcript",
        "seq": utt.seq,
        "text": text,
        "language": language,
        "started_at": iso(utt.ended_at - audio_seconds),
        "ended_at": iso(utt.ended_at),
        "audio_seconds": round(audio_seconds, 3),
        "inference_seconds": None if inference_seconds is None else round(inference_seconds, 4),
        "latency_seconds": round(finished_at - utt.ended_at, 3),
    }


def format_record(record: dict, mode: str) -> str:
    if mode == "jsonl":
        return json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    return f"ASR result: {record['text']}"
