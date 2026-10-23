import base64
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from voice_input import sensevoice_worker as w


class FakeEngine:
    languages = ["auto", "en"]

    def __init__(self):
        self.calls = []

    def transcribe(self, pcm, language):
        self.calls.append((len(pcm), language))
        if len(self.calls) == 2:
            raise RuntimeError("npu hiccup")
        return f" hello {len(self.calls)} "


def req(i, pcm=b"\x01\x00" * 160, **kw):
    return json.dumps({"id": i, "pcm": base64.b64encode(pcm).decode(), **kw}) + "\n"


class ServeTests(unittest.TestCase):
    def run_serve(self, lines, engine=None):
        out = io.StringIO()
        engine = engine or FakeEngine()
        rc = w.serve(engine, io.StringIO("".join(lines)), out)
        return rc, [json.loads(l) for l in out.getvalue().splitlines()], engine

    def test_engine_created_once_and_ordered_replies(self):
        rc, msgs, engine = self.run_serve(
            [req(1, language="en"), req(2), req(3, language="en"), '{"cmd": "shutdown"}\n', req(4)])
        self.assertEqual(rc, 0)
        self.assertEqual(msgs[0], {"event": "ready"})
        self.assertEqual([m["id"] for m in msgs[1:]], [1, 2, 3])
        self.assertEqual(msgs[1]["text"], "hello 1")
        self.assertFalse(msgs[2]["ok"])  # failure is reported and the loop continues
        self.assertIn("npu hiccup", msgs[2]["error"])
        self.assertTrue(msgs[3]["ok"])
        self.assertEqual(len(engine.calls), 3)
        self.assertAlmostEqual(msgs[1]["audio_seconds"], 0.01)

    def test_bad_requests_do_not_kill_worker(self):
        rc, msgs, _ = self.run_serve(
            ["not json\n", req(1, language="xx"), req(2, pcm=b""), req(3)])
        self.assertEqual([m.get("ok") for m in msgs[1:]], [False, False, False, True])

    def test_odd_byte_count_is_trimmed(self):
        _, msgs, engine = self.run_serve([req(1, pcm=b"\x01\x00\x02")])
        self.assertTrue(msgs[1]["ok"])
        self.assertEqual(engine.calls[0][0], 2)

    def test_default_model_root(self):
        self.assertEqual(w.default_model_root(Path("/sv/python"), "ax650"),
                         Path("/sv/python/models/SenseVoice/sensevoice_ax650"))

    def test_startup_failure_reports_event_and_exit_code(self):
        with tempfile.TemporaryDirectory() as d:
            proc = subprocess.run(
                [sys.executable, w.__file__, "--sensevoice-dir", d, "--chip", "ax650"],
                input="", capture_output=True, text=True, timeout=30)
        self.assertEqual(proc.returncode, w.STARTUP_FAILURE)
        msg = json.loads(proc.stdout.splitlines()[0])
        self.assertEqual(msg["event"], "error")
        self.assertIn("model files missing", msg["error"])
