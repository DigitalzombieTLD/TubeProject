import base64
from array import array
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest

from voice_input import cli, whisper_worker as ww
from voice_input.cli import build_parser, resolve_settings
from voice_input.config import Settings
from voice_input.transcriber import TranscriptionError, WhisperSession

PCM = array("h", [1000] * 1600).tobytes()


class FakeArray:
    def __init__(self, data):
        self.data = data

    def astype(self, _):
        return self

    def __truediv__(self, d):
        return FakeArray([x / d for x in self.data])


class FakeNp:
    float32 = "float32"

    @staticmethod
    def frombuffer(buf, dtype):
        return FakeArray(list(array("h", buf)))


class FakeModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, **kw):
        self.calls.append((len(audio.data), kw))
        if len(self.calls) == 2:
            raise RuntimeError("decode failed")
        return {"text": f" heard {len(self.calls)} "}


def fake_whisper():
    mod = types.SimpleNamespace(loads=[], model=FakeModel())

    def load_model(name, **kw):
        mod.loads.append((name, kw))
        return mod.model
    mod.load_model = load_model
    mod.tokenizer = types.SimpleNamespace(LANGUAGES={"en": "english", "de": "german"})
    return mod


def req(i, language="en"):
    return json.dumps({"id": i, "language": language, "pcm": base64.b64encode(PCM).decode()}) + "\n"


class EngineTests(unittest.TestCase):
    def test_model_loaded_once_for_many_utterances_with_recovery(self):
        mod = fake_whisper()
        engine = ww.WhisperEngine("base.en", whisper_module=mod, np=FakeNp)
        out = io.StringIO()
        rc = ww.serve(engine, io.StringIO(req(1) + req(2) + req(3, "auto") + req(4, "xx")), out)
        msgs = [json.loads(l) for l in out.getvalue().splitlines()]
        self.assertEqual(rc, 0)
        self.assertEqual(len(mod.loads), 1)
        self.assertEqual(mod.loads[0], ("base.en", {"device": "cpu"}))
        self.assertEqual(msgs[0], {"event": "ready"})
        self.assertEqual([m["ok"] for m in msgs[1:]], [True, False, True, False])
        self.assertEqual(msgs[1]["text"], "heard 1")
        self.assertIn("decode failed", msgs[2]["error"])
        self.assertIn("unsupported language", msgs[4]["error"])
        calls = mod.model.calls
        self.assertEqual(calls[0][0], 1600)
        self.assertEqual(calls[0][1]["language"], "en")
        self.assertFalse(calls[0][1]["fp16"])
        self.assertIsNone(calls[2][1]["language"])  # auto -> Whisper language detection

    def test_task_device_and_download_root_are_passed(self):
        mod = fake_whisper()
        engine = ww.WhisperEngine("small", "translate", "cuda", "/cache", whisper_module=mod, np=FakeNp)
        engine.transcribe(PCM, "de")
        self.assertEqual(mod.loads[0], ("small", {"device": "cuda", "download_root": "/cache"}))
        self.assertEqual(mod.model.calls[0][1]["task"], "translate")
        self.assertTrue(mod.model.calls[0][1]["fp16"])

    def test_worker_subprocess_reports_missing_whisper(self):
        with tempfile.TemporaryDirectory() as d:  # no 'whisper' importable
            proc = subprocess.run([sys.executable, ww.__file__, "--model", "base.en"],
                                  input="", capture_output=True, text=True, timeout=60, cwd=d)
        if proc.returncode == 0:
            self.skipTest("whisper is installed here")
        self.assertEqual(proc.returncode, ww.STARTUP_FAILURE)
        self.assertEqual(json.loads(proc.stdout.splitlines()[0])["event"], "error")


class FakeWhisperPackageSession(unittest.TestCase):
    """End to end through WhisperSession with fake whisper/numpy packages on the worker's path."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        (root / "whisper").mkdir()
        (root / "whisper" / "__init__.py").write_text(
            "import os\nfrom . import tokenizer\n"
            "class M:\n n = 0\n def transcribe(self, a, **k):\n  self.n += 1\n"
            "  if self.n == 2: raise RuntimeError('boom')\n"
            "  return {'text': '%d:%d:%s' % (os.getpid(), self.n, k['language'])}\n"
            "def load_model(name, **k):\n  if name == 'bad': raise RuntimeError('no weights')\n  return M()\n")
        (root / "whisper" / "tokenizer.py").write_text("LANGUAGES = {'en': 'english'}\n")
        (root / "numpy.py").write_text(
            "float32 = 'f'\nclass A:\n def astype(self, t): return self\n"
            " def __truediv__(self, d): return self\n"
            "def frombuffer(b, dtype): return A()\n")
        env = dict(os.environ, PYTHONPATH=str(root))
        self._env = os.environ.get("PYTHONPATH")
        os.environ["PYTHONPATH"] = env["PYTHONPATH"]
        self.addCleanup(self._restore)

    def _restore(self):
        if self._env is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = self._env

    def session(self, model="base.en"):
        s = WhisperSession(sys.executable, model, "en", startup_timeout=60, timeout=60)
        self.addCleanup(s.shutdown)
        return s

    def test_one_load_many_utterances_recovery_and_shutdown(self):
        s = self.session()
        s.start()
        a = s.transcribe(PCM).text
        with self.assertRaisesRegex(TranscriptionError, "Whisper inference failed.*boom"):
            s.transcribe(PCM)
        b = s.transcribe(PCM).text
        self.assertEqual(a.split(":")[0], b.split(":")[0])  # same process: model not reloaded
        self.assertEqual((a.split(":")[1], b.split(":")[1], a.split(":")[2]), ("1", "3", "en"))
        self.assertEqual(s.starts, 1)
        proc = s._proc
        s.shutdown()
        self.assertIsNotNone(proc.poll())

    def test_startup_failure_is_reported(self):
        with self.assertRaisesRegex(TranscriptionError, "Whisper failed to start.*no weights"):
            self.session("bad").start()


class ConfigTests(unittest.TestCase):
    def settings(self, *argv):
        return resolve_settings(build_parser().parse_args(["--engine", "whisper", *argv]))

    def test_defaults_english_model_without_sensevoice_main(self):
        s = self.settings()
        self.assertEqual((s.effective_whisper_model(), s.effective_whisper_language(), s.whisper_device),
                         ("base.en", "en", "cpu"))

    def test_multilingual_selection(self):
        s = self.settings("--whisper-language", "de")
        self.assertEqual(s.effective_whisper_model(), "base")
        s = self.settings("--whisper-model", "small", "--whisper-language", "auto", "--whisper-task", "translate")
        self.assertEqual((s.effective_whisper_model(), s.whisper_task), ("small", "translate"))

    def test_invalid_values(self):
        for argv in (["--whisper-model", "nope"], ["--whisper-language", "German"],
                     ["--whisper-model", "base.en", "--whisper-language", "de"],
                     ["--whisper-model", "small.en", "--whisper-task", "translate"]):
            with self.assertRaises(ValueError, msg=argv):
                self.settings(*argv)

    def test_sensevoice_still_validated(self):
        with self.assertRaises(ValueError):
            resolve_settings(build_parser().parse_args(["--sensevoice-main", "x", "--language", "de"]))
        with self.assertRaises(ValueError):
            resolve_settings(build_parser().parse_args([]))

    def test_session_command(self):
        s = self.settings("--whisper-download-root", "/c", "--whisper-python", "/w/py")
        t = cli.make_whisper_transcriber(s)
        self.assertEqual(t.command()[:2], ["/w/py", str(Path(ww.__file__))])
        self.assertIn("--download-root", t.command())
        self.assertEqual(t.command()[t.command().index("--model") + 1], "base.en")


class RunTests(unittest.TestCase):
    def test_whisper_run_prints_transcripts_and_labels(self):
        from test_cli_run import FakeCapture, FakeTranscriber, speech_then_silence
        out, err = io.StringIO(), io.StringIO()
        s = Settings(engine="whisper", silence=0.2, min_utterance=0.1, preroll=0.0, output="jsonl",
                     whisper_language="de", whisper_model="small")
        cli.run(s, FakeTranscriber(), FakeCapture(speech_then_silence(1)), out, err)
        rec = json.loads(out.getvalue().splitlines()[0])
        self.assertEqual((rec["text"], rec["language"]), ("utt1", "de"))
        self.assertIn("Loading Whisper model small on cpu", err.getvalue())
        self.assertIn("Whisper ready.", err.getvalue())


if __name__ == "__main__":
    unittest.main()
