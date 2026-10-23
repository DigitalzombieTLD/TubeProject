"""Command-line entry point: mic -> segmenter -> bounded queue -> SenseVoice worker."""
import argparse
from dataclasses import fields
import os
from pathlib import Path
import sys

from .audio import extract_channel, rms
from .capture import Capture
from .config import LANGUAGES, Settings, load_config
from .jobqueue import POLICIES, BoundedJobQueue, Worker
from .segmenter import Segmenter, duration_seconds
from .transcriber import SenseVoiceTranscriber

BLOCK_MS = 20


def build_parser() -> argparse.ArgumentParser:
    d = Settings()
    p = argparse.ArgumentParser(
        prog="tube-voice-input",
        description="Live XVF3800 microphone transcription with SenseVoice on AX650N.")
    p.add_argument("--config", help="JSON config file (see config.example.json); CLI flags override it")
    p.add_argument("--device", help=f"ALSA capture device (default {d.device})")
    p.add_argument("--channel", type=int, choices=(0, 1), help="stereo channel to use (default 0)")
    p.add_argument("--silence", type=float, help=f"seconds of silence ending an utterance ({d.silence})")
    p.add_argument("--threshold", type=float, help=f"RMS speech threshold ({d.threshold})")
    p.add_argument("--preroll", type=float, help=f"seconds kept before speech starts ({d.preroll})")
    p.add_argument("--min-utterance", type=float, help=f"minimum voiced seconds ({d.min_utterance})")
    p.add_argument("--max-utterance", type=float, help=f"force-split utterances at N seconds ({d.max_utterance})")
    p.add_argument("--queue-size", type=int, help=f"max utterances waiting for transcription ({d.queue_size})")
    p.add_argument("--queue-policy", choices=POLICIES, help=f"when full, drop this utterance ({d.queue_policy})")
    p.add_argument("--language", choices=LANGUAGES, help=f"SenseVoice language ({d.language})")
    p.add_argument("--chip", help=f"SenseVoice --chip value ({d.chip})")
    p.add_argument("--sensevoice-main", help="path to SenseVoice python/main.py")
    p.add_argument("--sensevoice-python", help="Python interpreter for SenseVoice (default: this one)")
    p.add_argument("--timeout", type=float, help=f"per-utterance inference timeout, seconds ({d.timeout})")
    return p


def resolve_settings(args) -> Settings:
    values = load_config(args.config) if args.config else {}
    for f in fields(Settings):
        v = getattr(args, f.name, None)
        if v is not None:
            values[f.name] = v
    s = Settings(**values)
    s.validate()
    return s


def run(s: Settings) -> int:
    main_py = Path(s.sensevoice_main).expanduser().resolve()
    if not main_py.is_file():
        print(f"error: SenseVoice main.py not found: {main_py}", file=sys.stderr)
        return 2
    python = os.path.expanduser(s.sensevoice_python) if s.sensevoice_python else sys.executable
    transcriber = SenseVoiceTranscriber(python, main_py, s.chip, s.language, s.timeout)

    def handle(mono):
        try:
            print(f"ASR result: {transcriber.transcribe(mono)}", flush=True)
        except Exception as exc:
            print(f"error: transcription failed: {exc}", file=sys.stderr, flush=True)

    jobs = BoundedJobQueue(
        s.queue_size, s.queue_policy,
        on_drop=lambda: print(
            f"warning: transcription queue full; utterance dropped ({s.queue_policy})",
            file=sys.stderr, flush=True))
    worker = Worker(jobs, handle, lambda e: print(f"error: worker: {e}", file=sys.stderr, flush=True))
    seg = Segmenter(BLOCK_MS / 1000, s.threshold, s.silence, s.preroll, s.min_utterance, s.max_utterance)
    cap = Capture(s.device, BLOCK_MS)

    def submit(mono):
        if mono is not None:
            if jobs.put(mono):
                print(f"[queued {duration_seconds(mono):.1f}s, pending {jobs.qsize()}]", flush=True)

    rc = 0
    drain = True
    try:
        cap.start()
        worker.start()
        print(f"Listening on {s.device}, channel {s.channel}. Ctrl-C to stop.", flush=True)
        for block in cap.blocks():
            mono = extract_channel(block, s.channel)
            submit(seg.feed(mono, rms(mono)))
    except KeyboardInterrupt:
        print("\nStopping...", flush=True)
        drain = False
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        rc = 1
    finally:
        cap.stop()
        if drain:
            submit(seg.flush())
        else:
            transcriber.shutdown()  # terminate any running inference right away
        if worker.started:
            worker.stop(drain=drain)
        transcriber.shutdown()
    return rc


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = resolve_settings(args)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return run(settings)


if __name__ == "__main__":
    raise SystemExit(main())
