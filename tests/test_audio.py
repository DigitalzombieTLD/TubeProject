from array import array
import unittest
import wave
import tempfile
from pathlib import Path

from voice_input.audio import extract_channel, rms, write_wav


def stereo(pairs):
    a = array("h")
    for l, r in pairs:
        a.extend([l, r])
    return a.tobytes()


class AudioTests(unittest.TestCase):
    def test_extract_channels(self):
        raw = stereo([(1, 10), (2, 20), (3, 30)])
        self.assertEqual(list(array("h", extract_channel(raw, 0))), [1, 2, 3])
        self.assertEqual(list(array("h", extract_channel(raw, 1))), [10, 20, 30])

    def test_extract_drops_partial_frame_and_validates(self):
        raw = stereo([(1, 2)]) + b"\x01\x00"
        self.assertEqual(len(extract_channel(raw, 0)), 2)
        with self.assertRaises(ValueError):
            extract_channel(raw, 2)

    def test_rms(self):
        self.assertEqual(rms(b""), 0.0)
        self.assertAlmostEqual(rms(array("h", [3, -3, 3, -3]).tobytes()), 3.0)

    def test_write_wav(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.wav"
            write_wav(p, array("h", [1, 2, 3]).tobytes())
            with wave.open(str(p)) as w:
                self.assertEqual((w.getnchannels(), w.getframerate(), w.getsampwidth(), w.getnframes()),
                                 (1, 16000, 2, 3))
