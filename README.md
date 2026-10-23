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
using a simple RMS threshold, and hands each utterance (mono S16_LE 16 kHz PCM) to a **persistent
SenseVoice worker process** that loaded the model once at startup. Capture never waits for inference:
utterances go into a bounded queue that one worker thread drains, in order.

```
arecord (stereo S16_LE 16 kHz) -> channel select -> RMS segmenter -> bounded queue -> ASR thread
      -> SenseVoiceSession --JSON Lines over pipes--> sensevoice_worker.py (SenseVoiceAx loaded once, AXCL)
      -> transcript output (text or JSON Lines) -> [future: LLM consumer]
```

### Persistent ASR architecture

Why: running `python main.py -i file` per utterance pays Python start-up, imports and AXCL/model
initialisation every time. The owner measured on the known sample (`../example/en.mp3`, 7.176 s of audio):

| What | Value |
| --- | --- |
| SenseVoice-reported `Latency` (model inference only) | ~0.135 s (RTF ~0.019) |
| Whole command, wall clock (`time -p`, `real`) | ~7.78 s (user 4.06 s, sys 1.15 s) |

So the reported latency is **inference only**; the end-to-end wall time of the CLI is dominated by the
process/model start-up around it (~7.6 s here), which is why per-utterance launches felt slower than real time.

What was verified from upstream source (the `SenseVoice` python package that `main.py` is in; read from the
public `ml-inory/sensevoice.axera` repository that this layout comes from, not from the owner's copy):

- `main.py` builds `SenseVoiceAx(model_path, cmvn_file, token_file, bpe_model, max_seq_len=256, beam_size=3,
  hot_words=None, streaming=False)` and calls `model.infer(path, language, print_rtf=True)`.
  `SenseVoiceAx.__init__` creates the `axengine.InferenceSession` (the `AXCLRTExecutionProvider` is picked by
  axengine); `infer()` also accepts `(waveform, sample_rate)`.
- Non-streaming `infer()` is stateless between calls: the front end builds a fresh `OnlineFbank` per call and
  `caches`/decoder state exist only in `streaming=True` mode. Hence no reset is needed between independent
  utterances, and one `SenseVoiceAx` can be reused for any number of them.
- Preprocessing: `main.py` loads files with `librosa.load(sr=16000)` (mono float32 in [-1, 1)). The worker feeds
  `int16 / 32768.0` as float32 at 16 kHz, which is the same representation; no resampling happens.
- `main.py` calls `download_model("SenseVoice")`, which may start a Hugging Face download. The worker instead
  uses the model directory directly and **never downloads**.

The `ax_asr_api` C++/Python binding was **not** used: its AXCL support and compatibility with a Debian 13
x86_64 AX-M1 host were not established, so this project does not depend on it.

Design: `SenseVoiceSession` (in `transcriber.py`) starts `sensevoice_worker.py` once with the
`--sensevoice-python` interpreter, waits for its `ready` message, then sends one request per utterance
(base64 PCM in a JSON line) and reads one reply, so jobs are processed strictly in order. The worker keeps
protocol traffic on a private copy of stdout and sends library log noise to stderr.

- **Startup errors** (missing model files, import failures, AXCL errors, timeout) are printed as
  `error: ...` and the program exits with status 2 before listening to the microphone.
- **Inference errors** are reported for that utterance only; the worker keeps running.
- **Crash or hang** (`--timeout`): the worker is killed and the next utterance starts a fresh one (one
  reload); this is the only time the model is reloaded.
- **Shutdown**: Ctrl-C stops `arecord`, aborts any waiting inference, closes the worker's stdin, then
  terminates/kills it and joins the reader thread. The worker runs in its own session so Ctrl-C is handled by
  this program only, and it exits by itself if this program dies (stdin EOF).
- **Memory** stays bounded by `queue_size` × `max_utterance` plus one in-flight request.

`--engine cli` keeps the old one-process-per-utterance behaviour as an explicit fallback; it reloads the model
for every utterance and exists only for comparison/debugging.

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

Against the existing installation (`~/ax-audio/SenseVoice`, `~/ax-audio/.venv312`), from this clone, with
machine-readable output for a later consumer (status lines go to stderr, transcripts to stdout):

```bash
source /etc/profile
PYTHONPATH=src ~/ax-audio/.venv312/bin/python -m voice_input \
  --sensevoice-main ~/ax-audio/SenseVoice/python/main.py \
  --sensevoice-python ~/ax-audio/.venv312/bin/python \
  --channel 0 --language en --output jsonl
```

The first start loads the model (watch for `SenseVoice ready.`); utterances after that do not reload it.
The SenseVoice model must already be present under `~/ax-audio/SenseVoice/python/models/SenseVoice/sensevoice_ax650/`
(created when you ran `main.py` before); if yours is elsewhere use `--sensevoice-model-dir`.

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
| `--sensevoice-model-dir` | `<main.py dir>/models/SenseVoice/sensevoice_<chip>` | Directory with `sensevoice.axmodel`, `am.mvn`, `tokens.txt`, `chn_jpn_yue_eng_ko_spectok.bpe.model` |
| `--engine` | `persistent` | `persistent` worker, or `cli` fallback (model reload per utterance) |
| `--output` | `text` | `text` or `jsonl` (see Output format) |
| `--startup-timeout` | `120` | Seconds to wait for the model to load |
| `--timeout` | `120` | Per-utterance inference timeout (s); on expiry the worker is restarted |

**Queue behaviour:** the capture loop never blocks. If the queue is full, one utterance is dropped according
to `--queue-policy` and a `warning:` is printed to stderr, so loss is never silent. Memory is bounded by
`queue_size` × `max_utterance`.

**Shutdown:** Ctrl-C stops `arecord`, stops the SenseVoice worker, and discards queued utterances. If
`arecord` exits on its own, queued utterances are finished before exit and the process returns a non-zero status.

### Output format

Default `--output text` prints `ASR result: <text>` per utterance (plus `[queued ...]` status lines on stdout).
Utterances with empty text are skipped.

`--output jsonl` prints exactly one JSON object per line on **stdout**; all status, warnings and errors go
to **stderr**:

```json
{"type":"transcript","seq":3,"text":"The tribal chieftain called for the boy.","language":"en","started_at":"2026-01-01T12:00:00.250Z","ended_at":"2026-01-01T12:00:07.426Z","audio_seconds":7.176,"inference_seconds":0.1354,"latency_seconds":0.41}
```

- `seq`: counted per segmented utterance; a gap means the bounded queue dropped one (also warned on stderr).
- `started_at`/`ended_at`: UTC; `ended_at` is when the segmenter closed the utterance, `started_at` is
  `ended_at - audio_seconds` (the audio includes pre-roll and the trailing silence).
- `inference_seconds`: model-reported, from the worker (`null` with `--engine cli`).
- `latency_seconds`: end of utterance until the transcript was ready (queue wait + inference + pipe round trip).
- New fields may be added; consumers should ignore unknown keys and check `type`.

A later LLM stage can simply read this stream line by line. No LLM integration exists in this repo yet.

### Hardware validation status

Honest status: this persistent implementation has **not** been exercised on the owner's physical microphone or
AX650N/AX-M1 NPU. Verified so far: unit tests with a test-double worker (no hardware), and the upstream-source
reading above. The AXCL/model load inside `sensevoice_worker.py` (`AxEngine`) is untested against real hardware
and the owner's SenseVoice copy, whose files may differ from the upstream source read. The owner's measurements
above are of the unchanged CLI, not of this worker. Expected but **unmeasured**: per-utterance wall time close to
the model's `Latency` once the model is loaded.
Please measure on the host: compare `inference_seconds` with `latency_seconds` in the JSONL output.

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
- `SenseVoice failed to start: ... model files missing`: the model directory does not contain the four files
  listed above; point `--sensevoice-model-dir` at it. The worker never downloads models.
- `error: transcription failed (utterance N): ...`: that utterance failed; the worker keeps going.
- `--engine cli` only: `no 'ASR result:' line` means run the same command shown in the error output manually.

### Limitations

- Prototype; non-streaming, whole-utterance inference (no partial text while speaking). Text appears after the
  silence pause plus inference.
- Energy-threshold segmentation only (no VAD), so noisy rooms need tuning.
- Parses the `ASR result:` line from SenseVoice's output; a different SenseVoice version may change it.
- Card selection, Unity integration, speaker/emotion recognition are not implemented or verified.
