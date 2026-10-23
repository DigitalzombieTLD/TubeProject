"""Test double speaking the sensevoice_worker.py protocol. text is '<pid>:<n>' so tests can see
whether the same process (same loaded 'model') served consecutive utterances."""
import argparse
import base64
import json
import os
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--sensevoice-dir")
p.add_argument("--chip")
p.add_argument("--model-dir", default="")
p.add_argument("--startup-fail", action="store_true")
p.add_argument("--fail-on", type=int, default=0)
p.add_argument("--crash-on", type=int, default=0)
p.add_argument("--hang-on", type=int, default=0)
args = p.parse_args()

out = os.fdopen(os.dup(1), "w")
os.dup2(2, 1)


def send(msg):
    out.write(json.dumps(msg) + "\n")
    out.flush()


if args.startup_fail:
    send({"event": "error", "error": "model files missing"})
    sys.exit(3)
print("noise that must not corrupt the protocol")
send({"event": "ready"})
n = 0
for line in sys.stdin:
    req = json.loads(line)
    if req.get("cmd") == "shutdown":
        break
    n += 1
    if n == args.crash_on:
        os._exit(9)
    if n == args.hang_on:
        time.sleep(60)
    if n == args.fail_on:
        send({"id": req["id"], "ok": False, "error": "boom"})
        continue
    pcm = base64.b64decode(req["pcm"])
    send({"id": req["id"], "ok": True, "text": f"{os.getpid()}:{n}:{req['language']}",
          "inference_seconds": 0.01, "audio_seconds": len(pcm) / 32000})
