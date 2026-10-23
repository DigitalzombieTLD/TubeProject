# TubeProject

TubeProject is a **multi-part project** that is being built up one component at a time.
It starts with **voice input**; everything else below is future scope.

| Part | Status |
| --- | --- |
| Voice input: XVF3800 microphone → SenseVoice (AX650N) speech-to-text | **Prototype (this repo)** |
| Unity API integration | Future scope |
| Speaker recognition | Future scope |
| Emotion recognition | Future scope |
| Other capabilities | Future scope |

## Part 1: live voice input

`src/voice_input/` captures audio from a Seeed reSpeaker XVF3800 with `arecord`, splits it into utterances
using a simple RMS threshold, and hands each utterance (mono, 16 kHz WAV) to the **existing SenseVoice
Python CLI** (`main.py`). Capture never waits for inference: utterances go into a bounded queue that
one worker thread drains.

```
arecord (stereo S16_LE 16 kHz) -> channel select -> RMS segmenter -> bounded queue -> worker -> SenseVoice main.py
```

### Verified environment (reported by the project owner)

- Debian 13 x86_64, two Radxa AX-M1 (AX650N) cards, both working with `axcl-smi` and `axcl_run_model`.
- Seeed reSpeaker XVF3800 USB array; ALSA device `plughw:CARD=Array,DEV=0`, 16 kHz, signed 16-bit stereo.
- Python 3.12 venv, `AXCLRTExecutionProvider`, model from `AXERA-TECH/SenseVoice`. This works:
  `python main.py -i ../example/en.mp3 --chip ax650 -l en` (English transcript, ~0.136 s for 7.176 s of audio).

Nothing else is claimed as verified. In particular, `--chip ax650` selects the model type, **not** a
specific card; which physical card is used has not been verified and is not controlled by this project.

### Setup (manual, host-specific)

This repository never downloads packages or runs `apt`/`dpkg`/`sudo`. Prepare the host yourself:

1. Install the AXCL runtime/drivers for the cards per Radxa/AXERA documentation and confirm `axcl-smi` works.
2. Install `alsa-utils` (for `arecord`) and confirm the mic: `arecord -L | grep -A1 Array`.
3. Obtain SenseVoice yourself (the AXERA-TECH/SenseVoice model package, which contains `python/main.py`
   and the AX650 model files). Model files are **not** part of this repo.
4. Create a SenseVoice Python environment following SenseVoice's own requirements. The tested one used
   Python 3.12 (Debian 13's default 3.13 had no matching NumPy wheel for `numpy<2`).
   Note: installing the `pyaxengine` (`axengine`) wheel upgraded NumPy to 2.x in the tested setup, while
   SenseVoice's requirements specify `numpy<2`. Check the versions afterwards
   (`python -m pip show numpy`), re-check SenseVoice's `requirements.txt`, and make sure `huggingface_hub`
   (imported by SenseVoice's `download_utils`) is installed in *that* venv. No pins are hardcoded here.
5. Run the unchanged SenseVoice test above and confirm it prints `ASR result:`.

The prototype itself only needs the Python standard library (Python ≥ 3.10). It can run from any
interpreter; SenseVoice is launched with the interpreter given by `--sensevoice-python`.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e .          # optional; or: PYTHONPATH=src python -m voice_input
```

### Running

```bash
source /etc/profile   # if your AXCL environment is set up there, as on the tested host
python -m voice_input \
  --sensevoice-main ~/ax-audio/SenseVoice/python/main.py \
  --sensevoice-python ~/ax-audio/.venv312/bin/python \
  --channel 0 --language en
```

Or copy `config.example.json`, edit the paths, and run `python -m voice_input --config my.json`
(command-line flags override the file). `python -m voice_input --help` lists every option:

| Option | Default | Meaning |
| --- | --- | --- |
| `--device` | `plughw:CARD=Array,DEV=0` | ALSA capture device (16 kHz, S16_LE, 2 ch) |
| `--channel` | `0` | Stereo channel to use (0 or 1) |
| `--silence` | `0.8` | Seconds of silence that end an utterance |
| `--threshold` | `500` | RMS speech threshold (lower if speech is missed) |
| `--preroll` | `0.2` | Seconds of audio kept before speech starts |
| `--min-utterance` | `0.3` | Minimum voiced seconds; shorter sounds are ignored |
| `--max-utterance` | `30` | Long speech is split at this length |
| `--queue-size` | `4` | Max utterances waiting for inference |
| `--queue-policy` | `drop-newest` | When full: `drop-newest` or `drop-oldest` |
| `--language` | `en` | `auto, zh, en, yue, ja, ko` |
| `--chip` | `ax650` | Passed to SenseVoice `--chip` |
| `--sensevoice-main` | *(required)* | Path to SenseVoice `python/main.py` |
| `--sensevoice-python` | current interpreter | Interpreter used for SenseVoice |
| `--timeout` | `120` | Per-utterance inference timeout (s) |

**Queue behaviour:** the capture loop never blocks. If the queue is full, one utterance is dropped according
to `--queue-policy` and a `warning:` is printed to stderr, so loss is never silent. Memory is bounded by
`queue_size` × `max_utterance`.

**Shutdown:** Ctrl-C stops `arecord`, terminates any running SenseVoice subprocess, and discards queued
utterances. If `arecord` exits on its own, queued utterances are finished before exit and the process
returns a non-zero status.

Each SenseVoice invocation is still one process per utterance (the model reloads each time); keeping the
model resident is a possible future improvement.

### Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests     # or: pytest
```

### Troubleshooting

- `[arecord] ...` lines are arecord's own stderr. `overrun!!!` means capture was not read fast enough;
  report it with the machine load. `audio open error`/`Device or resource busy`: check `arecord -L` and
  that nothing else holds the mic.
- No speech detected: lower `--threshold` (e.g. 250). Triggers on noise: raise it. Quiet or odd audio:
  try `--channel 1`.
- `SenseVoice main.py not found`: set `--sensevoice-main`.
- `ModuleNotFoundError` from SenseVoice: the packages are missing in the interpreter given by
  `--sensevoice-python`, not in this project's.
- `no 'ASR result:' line`: run the same command shown in the error output manually.
- Failures are printed as `error: transcription failed: ...` with SenseVoice's output; the worker keeps going.

### Limitations

- Prototype; non-streaming file-based inference, one SenseVoice process launch per utterance.
- Energy-threshold segmentation only (no VAD), so noisy rooms need tuning.
- Parses the `ASR result:` line from SenseVoice's output; a different SenseVoice version may change it.
- Card selection, Unity integration, speaker/emotion recognition are not implemented or verified.
