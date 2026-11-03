"""Regression check for the possession labeller: same flags, same labels, byte for byte.

    python tests/golden.py --check                 # run every variant, compare with golden_hashes.json
    python tests/golden.py --record                # re-record the hashes (after an INTENDED change)

Needs the data bundle (./data or $MTAH_DATA_DIR). What is stored is an md5 of each output CSV and the
per-half summary lines the run prints, never the labels themselves (the repo is public). The hashes
were recorded from the single-file best_so_far.py before it was split into starter/possession/.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPT = os.path.join(ROOT, "starter", "best_so_far.py")
HASHES = os.path.join(HERE, "golden_hashes.json")

VARIANTS = [
    [], ["--no-backdate"], ["--no-pass-by"], ["--no-goal-kick"], ["--no-free-kicks"],
    ["--no-slow-stop"], ["--no-keepers"], ["--dead-lead", "0"], ["--dead-tail", "0"],
    ["--shape-veto", "0"], ["--merge-gap", "0"], ["--slow-back", "0"], ["--nearest-radius", "1.0"],
    ["--shape-max", "8", "--shape-veto", "4"], ["--trace", "11185", "11300"],
]


def run(flags):
    """(csv name, csv md5, stdout lines that do not depend on the output folder)."""
    with tempfile.TemporaryDirectory() as out:
        done = subprocess.run([sys.executable, SCRIPT] + flags + ["--out", out], cwd=ROOT,
                              capture_output=True, text=True)
        if done.returncode:
            sys.exit("FAILED {}:\n{}".format(flags, done.stderr))
        (name,) = [f for f in os.listdir(out) if f.endswith(".csv")]
        with open(os.path.join(out, name), "rb") as handle:
            digest = hashlib.md5(handle.read()).hexdigest()
    lines = [line for line in done.stdout.splitlines()
             if line.strip() and ".csv" not in line and not line.startswith("score it")]
    return name, digest, lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--record", action="store_true")
    args = parser.parse_args()

    results = {" ".join(flags) or "(default)": run(flags) for flags in VARIANTS}
    if args.record:
        with open(HASHES, "w") as handle:
            json.dump({k: {"csv": v[0], "md5": v[1], "stdout": v[2]} for k, v in results.items()},
                      handle, indent=1)
        print("recorded {} variants in {}".format(len(results), os.path.relpath(HASHES, ROOT)))
        return
    with open(HASHES) as handle:
        golden = json.load(handle)
    bad = 0
    for key, (name, digest, lines) in results.items():
        want = golden.get(key)
        problems = []
        if want is None:
            problems.append("not in golden_hashes.json")
        else:
            if name != want["csv"]:
                problems.append("file name {} != {}".format(name, want["csv"]))
            if digest != want["md5"]:
                problems.append("labels differ")
            if lines != want["stdout"]:
                problems.append("printed summary differs")
        print("{:<5} {}{}".format("ok" if not problems else "FAIL", key,
                                  "" if not problems else "  -> " + "; ".join(problems)))
        bad += bool(problems)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
