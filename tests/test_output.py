import json
import unittest

from voice_input.output import Utterance, format_record, iso, transcript_record


class OutputTests(unittest.TestCase):
    def rec(self, inference=0.135):
        return transcript_record(Utterance(7, b"", 1000.0), "héllo", "en", 2.0, inference, 1001.5)

    def test_record_fields(self):
        r = self.rec()
        self.assertEqual(r["type"], "transcript")
        self.assertEqual((r["seq"], r["text"], r["language"]), (7, "héllo", "en"))
        self.assertEqual(r["ended_at"], "1970-01-01T00:16:40.000Z")
        self.assertEqual(r["started_at"], "1970-01-01T00:16:38.000Z")
        self.assertEqual((r["audio_seconds"], r["inference_seconds"], r["latency_seconds"]),
                         (2.0, 0.135, 1.5))

    def test_jsonl_is_one_parseable_line(self):
        line = format_record(self.rec(None), "jsonl")
        self.assertNotIn("\n", line)
        self.assertIn("héllo", line)
        self.assertIsNone(json.loads(line)["inference_seconds"])

    def test_text_default_keeps_legacy_format(self):
        self.assertEqual(format_record(self.rec(), "text"), "ASR result: héllo")

    def test_iso(self):
        self.assertTrue(iso(0).endswith("Z"))
