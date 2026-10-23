"""Command-line entry point: mic -> segmenter -> bounded queue -> SenseVoice worker."""
import argparse
from dataclasses import fields
import os
from pathlib import Path
import sys
import time

from .audio import extract_channel, rms
from .capture import Capture
from .config import ENGINES, LANGUAGES, OUTPUTS, Settings, load_config
from .jobqueue import POLICIES, BoundedJobQueue, Worker
from .output import Utterance, format_record, transcript_record
from .segmenter import Segmenter, duration_seconds
from .transcriber import SenseVoiceCliTranscriber, SenseVoiceSession, TranscriptionError

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
    p.add_argument("--sensevoice-model-dir",
                   help="directory with sensevoice.axmodel, am.mvn, tokens.txt, *.bpe.model "
                        "(default: <main.py dir>/models/SenseVoice/sensevoice_<chip>)")
    p.add_argument("--engine", choices=ENGINES,
                   help=f"persistent = model loaded once in a long-lived worker; cli = legacy "
                        f"one-process-per-utterance fallback ({d.engine})")
    p.add_argument("--output", choices=OUTPUTS,
                   help=f"text = 'ASR result: ...' lines; jsonl = one JSON object per transcript ({d.output})")
    p.add_argument("--startup-timeout", type=float, help=f"seconds to wait for the model to load ({d.startup_timeout})")
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


def make_transcriber(s: Settings, python: str, main_py: Path):
    if s.engine == "cli":
        return SenseVoiceCliTranscriber(python, main_py, s.chip, s.language, s.timeout)
    return SenseVoiceSession(python, main_py.parent, s.chip, s.language, s.timeout,
                             s.startup_timeout, s.sensevoice_model_dir)


def run(s: Settings, transcriber=None, capture=None, out=None, err=None) -> int:
    out = out or sys.stdout
    err = err or sys.stderr
    # In jsonl mode stdout carries only transcript records; everything else goes to stderr.
    info = err if s.output == "jsonl" else out

    def say(msg, stream=info):
        print(msg, file=stream, flush=True)

    if transcriber is None:
        main_py = Path(s.sensevoice_main).expanduser().resolve()
        if not main_py.is_file():
            say(f"error: SenseVoice main.py not found: {main_py}", err)
            return 2
        python = os.path.expanduser(s.sensevoice_python) if s.sensevoice_python else sys.executable
        transcriber = make_transcriber(s, python, main_py)

    if hasattr(transcriber, "start"):
        say(f"Loading SenseVoice ({s.chip}, language {s.language})...")
        try:
            transcriber.start()
        except TranscriptionError as exc:
            say(f"error: {exc}", err)
            transcriber.shutdown()
            return 2
        except KeyboardInterrupt:
            transcriber.shutdown()
            say("\nInterrupted during startup.", err)
            return 130
        say("SenseVoice ready.")

    def handle(utt):
        try:
            res = transcriber.transcribe(utt.pcm)
        except Exception as exc:
            say(f"error: transcription failed (utterance {utt.seq}): {exc}", err)
            return
        if not res.text:
            return
        record = transcript_record(utt, res.text, s.language, duration_seconds(utt.pcm),
                                   res.inference_seconds, time.time())
        say(format_record(record, s.output), out)

    jobs = BoundedJobQueue(
        s.queue_size, s.queue_policy,
        on_drop=lambda: say(f"warning: transcription queue full; utterance dropped ({s.queue_policy})", err))
    worker = Worker(jobs, handle, lambda e: say(f"error: worker: {e}", err))
    seg = Segmenter(BLOCK_MS / 1000, s.threshold, s.silence, s.preroll, s.min_utterance, s.max_utterance)
    cap = capture or Capture(s.device, BLOCK_MS)
    seq = 0

    def submit(mono):
        nonlocal seq
        if mono is not None:
            seq += 1
            if jobs.put(Utterance(seq, mono, time.time())):
                say(f"[queued {duration_seconds(mono):.1f}s, pending {jobs.qsize()}]")

    rc = 0
    drain = True
    try:
        cap.start()
        worker.start()
        say(f"Listening on {s.device}, channel {s.channel}. Ctrl-C to stop.")
        for block in cap.blocks():
            mono = extract_channel(block, s.channel)
            submit(seg.feed(mono, rms(mono)))
    except KeyboardInterrupt:
        say("\nStopping...")
        drain = False
    except RuntimeError as exc:
        say(f"error: {exc}", err)
        rc = 1
    finally:
        try:
            cap.stop()
            if drain:
                submit(seg.flush())
            else:
                transcriber.shutdown()  # abort any running inference right away
            if worker.started:
                worker.stop(drain=drain)
        finally:
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
