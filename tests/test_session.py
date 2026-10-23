from array import array
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest

from voice_input.transcriber import SenseVoiceSession, TranscriptionError

import sys
FAKE = Path(__file__).with_name("fake_worker.py")
PCM = array("h", [100] * 1600).tobytes()


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class SessionTests(unittest.TestCase):
    def session(self, *extra, **kw):
        s = SenseVoiceSession(sys.executable, tempfile.gettempdir(), "ax650", "en",
                              worker_script=FAKE, extra_args=list(extra), **kw)
        self.addCleanup(s.shutdown)
        return s

    def test_repeated_utterances_use_one_initialisation(self):
        s = self.session()
        s.start()
        results = [s.transcribe(PCM).text for _ in range(5)]
        pids = {r.split(":")[0] for r in results}
        self.assertEqual(len(pids), 1)
        self.assertEqual([r.split(":")[1] for r in results], ["1", "2", "3", "4", "5"])
        self.assertTrue(results[0].endswith(":en"))
        self.assertEqual(s.starts, 1)

    def test_lazy_start_and_inference_seconds(self):
        s = self.session()
        r = s.transcribe(PCM)
        self.assertEqual(r.inference_seconds, 0.01)
        self.assertEqual(s.starts, 1)

    def test_startup_error_is_reported(self):
        s = self.session("--startup-fail")
        with self.assertRaisesRegex(TranscriptionError, "model files missing"):
            s.start()

    def test_missing_interpreter(self):
        s = SenseVoiceSession("/nonexistent/python", tempfile.gettempdir(), worker_script=FAKE)
        with self.assertRaisesRegex(TranscriptionError, "cannot launch|No such"):
            s.start()

    def test_inference_failure_keeps_worker(self):
        s = self.session("--fail-on", "2")
        first = s.transcribe(PCM).text
        with self.assertRaisesRegex(TranscriptionError, "boom"):
            s.transcribe(PCM)
        third = s.transcribe(PCM).text
        self.assertEqual(first.split(":")[0], third.split(":")[0])
        self.assertEqual(s.starts, 1)

    def test_crash_recovers_with_new_worker(self):
        s = self.session("--crash-on", "2")
        first = s.transcribe(PCM).text
        with self.assertRaisesRegex(TranscriptionError, "exited unexpectedly"):
            s.transcribe(PCM)
        again = s.transcribe(PCM).text
        self.assertNotEqual(first.split(":")[0], again.split(":")[0])
        self.assertEqual(s.starts, 2)

    def test_hang_times_out_and_is_killed(self):
        s = self.session("--hang-on", "1", timeout=0.5)
        s.start()
        pid = s._proc.pid
        with self.assertRaisesRegex(TranscriptionError, "timed out"):
            s.transcribe(PCM)
        self.assertFalse(pid_alive(pid))

    def test_shutdown_stops_process_and_threads(self):
        s = self.session()
        s.start()
        pid, reader = s._proc.pid, s._reader
        s.shutdown()
        self.assertFalse(pid_alive(pid))
        self.assertFalse(reader.is_alive())
        with self.assertRaisesRegex(TranscriptionError, "shut down"):
            s.transcribe(PCM)
        s.shutdown()  # idempotent

    def test_shutdown_interrupts_running_inference(self):
        s = self.session("--hang-on", "1", timeout=30)
        s.start()
        pid = s._proc.pid
        errors = []

        def run():
            try:
                s.transcribe(PCM)
            except TranscriptionError as exc:
                errors.append(str(exc))

        t = threading.Thread(target=run)
        t.start()
        time.sleep(0.3)
        began = time.monotonic()
        s.shutdown()
        t.join(10)
        self.assertFalse(t.is_alive())
        self.assertLess(time.monotonic() - began, 10)
        self.assertEqual(errors, ["transcriber is shut down"])
        self.assertFalse(pid_alive(pid))

    def test_empty_audio_rejected(self):
        with self.assertRaisesRegex(TranscriptionError, "empty"):
            self.session().transcribe(b"")
