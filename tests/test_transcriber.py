from array import array
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from voice_input.transcriber import (SenseVoiceTranscriber, TranscriptionError,
                                     build_command, parse_result)

PCM = array("h", [100] * 1600).tobytes()


class TranscriberTests(unittest.TestCase):
    def test_command_uses_verified_options(self):
        self.assertEqual(build_command("py", Path("m.py"), Path("a.wav"), "ax650", "en"),
                         ["py", "m.py", "-i", "a.wav", "--chip", "ax650", "-l", "en"])

    def test_parse(self):
        self.assertEqual(parse_result("x\nASR result: hi there\n"), "hi there")
        self.assertIsNone(parse_result("nothing"))

    def _fake(self, body):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        p = Path(d.name) / "main.py"
        p.write_text(textwrap.dedent(body))
        return p

    def test_success_and_failure(self):
        ok = self._fake("import sys\nassert sys.argv[1]=='-i'\nprint('ASR result: hello')")
        self.assertEqual(SenseVoiceTranscriber(sys.executable, ok).transcribe(PCM), "hello")
        bad = self._fake("import sys\nprint('bad')\nsys.exit(3)")
        with self.assertRaisesRegex(TranscriptionError, "exited with 3"):
            SenseVoiceTranscriber(sys.executable, bad).transcribe(PCM)

    def test_timeout(self):
        slow = self._fake("import time\ntime.sleep(30)")
        with self.assertRaisesRegex(TranscriptionError, "timed out"):
            SenseVoiceTranscriber(sys.executable, slow, timeout=0.5).transcribe(PCM)
