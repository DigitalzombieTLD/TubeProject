import unittest

from voice_input.segmenter import Segmenter


def run(seg, pattern):
    """pattern: string of 'v' (voiced) / '.' (silent); block bytes are the index."""
    out = []
    for i, c in enumerate(pattern):
        r = seg.feed(bytes([i]), 1000.0 if c == "v" else 0.0)
        if r is not None:
            out.append(r)
    return out


def make(**kw):
    d = dict(block_seconds=1, threshold=500, silence=2, preroll=1, min_utterance=2, max_utterance=100)
    d.update(kw)
    return Segmenter(**d)


class SegmenterTests(unittest.TestCase):
    def test_utterance_with_preroll_and_trimmed_tail(self):
        out = run(make(), "..vvv..")
        self.assertEqual(out, [bytes([1, 2, 3, 4])])

    def test_short_sound_ignored(self):
        self.assertEqual(run(make(), "..v..."), [])

    def test_brief_pause_does_not_split(self):
        out = run(make(), "vv.vv..")
        self.assertEqual(out, [bytes([0, 1, 2, 3, 4])])

    def test_max_utterance_splits(self):
        out = run(make(max_utterance=3, silence=5, min_utterance=1), "vvvv")
        self.assertEqual(out, [bytes([0, 1, 2])])

    def test_flush(self):
        seg = make()
        run(seg, "vvv")
        self.assertEqual(seg.flush(), bytes([0, 1, 2]))
        self.assertIsNone(seg.flush())
