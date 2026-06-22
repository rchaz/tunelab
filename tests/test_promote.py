#!/usr/bin/env python3
"""Evidence tests for skills/tune-loop/scripts/promote.py.

Invokes the real script via subprocess. Prints one 'PASS: <check>' line per
check; exits non-zero on the first failure. promote.py is stdlib-only.
"""

import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "skills", "tune-loop", "scripts", "promote.py")


def run(args):
    return subprocess.run(["python3", SCRIPT] + args, capture_output=True, text=True)


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f)
    return path


def main():
    tmp = tempfile.mkdtemp(prefix="tunelab-promote-test-")
    champ = write_json(os.path.join(tmp, "champ.json"),
                       {"accuracy": 0.80, "cost_per_1k": 1.2, "n": 200})
    chal_win = write_json(os.path.join(tmp, "chal_win.json"),
                          {"accuracy": 0.88, "cost_per_1k": 0.3, "n": 200})
    chal_tie = write_json(os.path.join(tmp, "chal_tie.json"),
                          {"accuracy": 0.805, "cost_per_1k": 0.3, "n": 200})

    # 1. Challenger clears bar AND beats champion by margin -> PROMOTE (exit 0).
    led1 = os.path.join(tmp, "led1.txt")
    r = run(["--champion", champ, "--challenger", chal_win, "--bar", "0.85",
             "--min-margin", "0.02", "--metric", "accuracy",
             "--slice-id", "s-promote", "--ledger", led1])
    if r.returncode != 0:
        fail(f"clear-win promote exited {r.returncode}: {r.stderr}")
    if "DECISION: PROMOTE" not in r.stdout:
        fail(f"expected PROMOTE in output, got: {r.stdout!r}")
    if "s-promote" not in open(led1).read():
        fail("consumed slice was not recorded in the ledger after adjudication")
    print("PASS: challenger clears bar + beats margin -> PROMOTE (exit 0), slice recorded")

    # 2. Within-noise-band win (margin 0.005 < min-margin 0.02) -> RETAIN (exit 0).
    led2 = os.path.join(tmp, "led2.txt")
    r = run(["--champion", champ, "--challenger", chal_tie, "--bar", "0.80",
             "--min-margin", "0.02", "--metric", "accuracy",
             "--slice-id", "s-retain", "--ledger", led2])
    if r.returncode != 0:
        fail(f"noise-band retain exited {r.returncode}: {r.stderr}")
    if "DECISION: RETAIN" not in r.stdout:
        fail(f"noise-band win should RETAIN champion, got: {r.stdout!r}")
    print("PASS: noise-band win (margin < min-margin) -> RETAIN (exit 0)")

    # 3. Descriptor version bumps only on PROMOTE.
    desc_in = write_json(os.path.join(tmp, "descriptor.json"), {"version": 3, "name": "router"})
    desc_out = os.path.join(tmp, "descriptor_out.json")
    led3 = os.path.join(tmp, "led3.txt")
    r = run(["--champion", champ, "--challenger", chal_win, "--bar", "0.85",
             "--min-margin", "0.02", "--metric", "accuracy",
             "--slice-id", "s-desc", "--ledger", led3,
             "--descriptor-in", desc_in, "--descriptor-out", desc_out])
    if r.returncode != 0:
        fail(f"promote-with-descriptor exited {r.returncode}: {r.stderr}")
    d = json.load(open(desc_out))
    if d.get("version") != 4 or d.get("_promoted_from_slice") != "s-desc":
        fail(f"descriptor not bumped/annotated on promote: {d}")
    print("PASS: PROMOTE bumps descriptor version 3 -> 4 and records the slice")

    # 3b. --challenger-descriptor: on PROMOTE the challenger becomes the new champion
    # (the record says WHAT won), with version continuing the champion's lineage.
    chal_desc = write_json(os.path.join(tmp, "challenger_desc.json"),
                           {"version": 1, "name": "router", "train": "lora:runs/x/adapters"})
    champ_desc = write_json(os.path.join(tmp, "champ_desc.json"),
                            {"version": 5, "name": "router", "train": "none (base)"})
    desc_out2 = os.path.join(tmp, "descriptor_out2.json")
    r = run(["--champion", champ, "--challenger", chal_win, "--bar", "0.85",
             "--min-margin", "0.02", "--metric", "accuracy",
             "--slice-id", "s-chaldesc", "--ledger", os.path.join(tmp, "led3b.txt"),
             "--descriptor-in", champ_desc, "--challenger-descriptor", chal_desc,
             "--descriptor-out", desc_out2])
    if r.returncode != 0:
        fail(f"promote-with-challenger-descriptor exited {r.returncode}: {r.stderr}")
    d = json.load(open(desc_out2))
    if d.get("train") != "lora:runs/x/adapters":
        fail(f"promoted descriptor should be the CHALLENGER's architecture, got: {d}")
    if d.get("version") != 6:  # max(challenger 1, champion 5) + 1
        fail(f"promoted version should continue champion lineage (6), got {d.get('version')}")
    if d.get("_promoted_from_slice") != "s-chaldesc":
        fail(f"promoted descriptor missing provenance: {d}")
    print("PASS: --challenger-descriptor -> challenger becomes new champion; version 5 -> 6")

    # 4. Reusing a consumed slice id -> discipline violation (exit 2).
    r = run(["--champion", champ, "--challenger", chal_win, "--bar", "0.85",
             "--min-margin", "0.02", "--metric", "accuracy",
             "--slice-id", "s-promote", "--ledger", led1])
    if r.returncode != 2:
        fail(f"slice reuse should exit 2, got {r.returncode}")
    if "already consumed" not in r.stderr:
        fail(f"slice reuse did not name the violation: {r.stderr}")
    print("PASS: reusing a consumed slice id -> discipline violation (exit 2)")

    # 5. Missing metric in an eval json -> clean exit 2, no traceback.
    bad_metric = write_json(os.path.join(tmp, "no_metric.json"), {"f1": 0.9, "n": 200})
    r = run(["--champion", champ, "--challenger", bad_metric, "--bar", "0.85",
             "--metric", "accuracy", "--slice-id", "s-nometric",
             "--ledger", os.path.join(tmp, "led5.txt")])
    if r.returncode != 2:
        fail(f"missing metric should exit 2, got {r.returncode}: {r.stderr}")
    if "Traceback" in r.stderr:
        fail(f"missing metric crashed with a traceback: {r.stderr}")
    print("PASS: missing --metric in eval json -> clean exit 2, no traceback")

    # 6. Missing eval file (the most likely fat-finger) -> clean exit 2, no traceback.
    for which, args in (
        ("champion", ["--champion", os.path.join(tmp, "nope.json"), "--challenger", chal_win]),
        ("challenger", ["--champion", champ, "--challenger", os.path.join(tmp, "nope.json")]),
    ):
        r = run(args + ["--bar", "0.85", "--metric", "accuracy",
                        "--slice-id", f"s-missing-{which}",
                        "--ledger", os.path.join(tmp, f"led6{which}.txt")])
        if r.returncode != 2:
            fail(f"missing {which} file should exit 2, got {r.returncode}: {r.stderr}")
        if "Traceback" in r.stderr:
            fail(f"missing {which} file crashed with a traceback: {r.stderr}")
        if "not found" not in r.stderr or which not in r.stderr:
            fail(f"missing {which} file lacked a clean naming message: {r.stderr}")
        print(f"PASS: missing {which} eval file -> clean exit 2 naming it, no traceback")

    # 7. Malformed JSON eval file -> clean exit 2, no traceback.
    broken = os.path.join(tmp, "broken.json")
    with open(broken, "w") as f:
        f.write("{not valid json")
    r = run(["--champion", broken, "--challenger", chal_win, "--bar", "0.85",
             "--metric", "accuracy", "--slice-id", "s-broken",
             "--ledger", os.path.join(tmp, "led7.txt")])
    if r.returncode != 2:
        fail(f"malformed JSON should exit 2, got {r.returncode}: {r.stderr}")
    if "Traceback" in r.stderr:
        fail(f"malformed JSON crashed with a traceback: {r.stderr}")
    if "not valid JSON" not in r.stderr:
        fail(f"malformed JSON lacked a clean message: {r.stderr}")
    print("PASS: malformed JSON eval file -> clean exit 2, no traceback")

    print("\nALL CHECKS PASSED (test_promote)")


if __name__ == "__main__":
    main()
