from array import array
import io
import json
import unittest

from voice_input import cli
from voice_input.config import Settings
from voice_input.transcriber import Transcription, TranscriptionError

LOUD = array("h", [3000] * 320).tobytes() * 2   # one 20 ms stereo block (320 frames)
QUIET = bytes(len(LOUD))


class FakeCapture:
    def __init__(self, blocks, interrupt=False):
        self.blocks_, self.interrupt, self.stopped = blocks, interrupt, False

    def start(self):
        pass

    def blocks(self):
        yield from self.blocks_
        if self.interrupt:
            raise KeyboardInterrupt
        raise RuntimeError("arecord stopped")

    def stop(self):
        self.stopped = True


class FakeTranscriber:
    def __init__(self, fail_start=False, fail_on=0):
        self.fail_start, self.fail_on = fail_start, fail_on
        self.started = self.shutdowns = self.calls = 0

    def start(self):
        if self.fail_start:
            raise TranscriptionError("cannot start")
        self.started += 1

    def transcribe(self, pcm):
        self.calls += 1
        if self.calls == self.fail_on:
            raise TranscriptionError("boom")
        return Transcription(f"utt{self.calls}", 0.1)

    def shutdown(self):
        self.shutdowns += 1


def speech_then_silence(n_utts):
    blocks = []
    for _ in range(n_utts):
        blocks += [LOUD] * 30 + [QUIET] * 10
    return blocks


def settings(**kw):
    base = dict(sensevoice_main="x", silence=0.2, min_utterance=0.1, preroll=0.0)
    base.update(kw)
    return Settings(**base)


class RunTests(unittest.TestCase):
    def go(self, s, transcriber, capture):
        out, err = io.StringIO(), io.StringIO()
        rc = cli.run(s, transcriber, capture, out, err)
        return rc, out.getvalue(), err.getvalue()

    def test_text_mode_processes_utterances_in_order_with_single_start(self):
        t = FakeTranscriber()
        rc, out, err = self.go(settings(), t, FakeCapture(speech_then_silence(3)))
        self.assertEqual(rc, 1)  # fake capture ends with an error, like arecord dying
        lines = [l for l in out.splitlines() if l.startswith("ASR result:")]
        self.assertEqual(lines, ["ASR result: utt1", "ASR result: utt2", "ASR result: utt3"])
        self.assertEqual(t.started, 1)
        self.assertGreaterEqual(t.shutdowns, 1)

    def test_jsonl_mode_keeps_stdout_machine_readable(self):
        t = FakeTranscriber()
        _, out, err = self.go(settings(output="jsonl"), t, FakeCapture(speech_then_silence(2)))
        records = [json.loads(l) for l in out.splitlines()]
        self.assertEqual([r["text"] for r in records], ["utt1", "utt2"])
        self.assertEqual([r["seq"] for r in records], [1, 2])
        self.assertIn("Listening", err)
        self.assertIn("[queued", err)

    def test_inference_failure_is_recoverable(self):
        t = FakeTranscriber(fail_on=1)
        _, out, err = self.go(settings(), t, FakeCapture(speech_then_silence(2)))
        self.assertIn("utt2", out)
        self.assertIn("transcription failed (utterance 1): boom", err)

    def test_startup_error_is_clean(self):
        t = FakeTranscriber(fail_start=True)
        cap = FakeCapture([])
        rc, out, err = self.go(settings(), t, cap)
        self.assertEqual(rc, 2)
        self.assertIn("error: cannot start", err)
        self.assertEqual(t.shutdowns, 1)
        self.assertFalse(cap.stopped)  # capture never started

    def test_ctrl_c_shuts_down_without_draining(self):
        t = FakeTranscriber()
        cap = FakeCapture(speech_then_silence(1)[:20], interrupt=True)
        rc, out, _ = self.go(settings(), t, cap)
        self.assertEqual(rc, 0)
        self.assertTrue(cap.stopped)
        self.assertIn("Stopping", out)
        self.assertGreaterEqual(t.shutdowns, 2)

    def test_empty_transcripts_are_not_emitted(self):
        class Empty(FakeTranscriber):
            def transcribe(self, pcm):
                return Transcription("", 0.1)
        _, out, _ = self.go(settings(), Empty(), FakeCapture(speech_then_silence(1)))
        self.assertNotIn("ASR result", out)
