import unittest

from voice_input.cli import build_parser, resolve_settings


class ConfigTests(unittest.TestCase):
    def test_cli_overrides_defaults(self):
        a = build_parser().parse_args(["--sensevoice-main", "/x/main.py", "--channel", "1"])
        s = resolve_settings(a)
        self.assertEqual((s.channel, s.language, s.chip, s.device),
                         (1, "en", "ax650", "plughw:CARD=Array,DEV=0"))

    def test_main_required(self):
        with self.assertRaises(ValueError):
            resolve_settings(build_parser().parse_args([]))
