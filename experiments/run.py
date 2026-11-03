"""Run one labeller configuration, score it, and append the result to experiments/log.jsonl.

    python experiments/run.py NAME [--note "why"] [--gt public|PATH] [--bootstrap 1000] \
        [--script starter/best_so_far.py] -- [labeller flags ...]

Example:
    python experiments/run.py best_so_far --note "reference"
    python experiments/run.py touch1.2 -- --nearest-radius 1.2

Everything after `--` goes to the labeller unchanged. The labels go to experiments/out/NAME.csv
(gitignored: a full label file is a copy of the answer). The log line holds only metrics, the
flags, the git commit and a hash of the label file, so it is safe to commit and to diff.

    python experiments/run.py --show        # the log as a table, newest last
"""

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
import run_evaluation as ev  # noqa: E402

LOG = os.path.join(HERE, "log.jsonl")
OUT = os.path.join(HERE, "out")


def git(*args):
    try:
        return subprocess.check_output(("git",) + args, cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def summary(result):
    t12 = result["transitions"][min(result["transitions"])]
    return {
        "spellF1": round(result["spell_f1_mean"], 4),
        "spell_ci": [round(result["spell_bootstrap_low"], 4), round(result["spell_bootstrap_high"], 4)],
        "macroF1": round(result["macro_f1"], 4),
        "macro_ci": [round(result["bootstrap_low"], 4), round(result["bootstrap_high"], 4)],
        "transF1": round(t12["f1"], 4),
        "n_pred_spells": result["n_pred_spells"], "n_gt_spells": result["n_gt_spells"],
        "team_acc_both_alive": round(result["team_acc_both_alive"], 4),
    }


def show():
    if not os.path.isfile(LOG):
        print("no log yet")
        return
    print("{:<22} {:>7} {:>16} {:>7} {:>7} {:>5}  {}".format(
        "name", "spellF1", "spell 95% CI", "macro", "transF", "spl", "flags / note"))
    with open(LOG) as handle:
        for line in handle:
            r = json.loads(line)
            m = r["metrics"]
            print("{:<22} {:>7.4f} {:>16} {:>7.4f} {:>7.4f} {:>2}/{:<2}  {} {}".format(
                r["name"][:22], m["spellF1"], "{:.3f}-{:.3f}".format(*m["spell_ci"]), m["macroF1"],
                m["transF1"], m["n_pred_spells"], m["n_gt_spells"], " ".join(r["flags"]),
                ("# " + r["note"]) if r["note"] else ""))


def main():
    argv = sys.argv[1:]
    flags = []
    if "--" in argv:
        cut = argv.index("--")
        argv, flags = argv[:cut], argv[cut + 1:]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", nargs="?")
    parser.add_argument("--note", default="")
    parser.add_argument("--gt", default="public")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--script", default=os.path.join("starter", "best_so_far.py"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)
    if args.show:
        return show()
    if not args.name:
        parser.error("a name is required")

    out_dir = os.path.join(OUT, args.name)
    os.makedirs(out_dir, exist_ok=True)
    for old in os.listdir(out_dir):
        if old.endswith(".csv"):
            os.remove(os.path.join(out_dir, old))
    cmd = [sys.executable, os.path.join(ROOT, args.script)] + flags + ["--out", out_dir]
    subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    produced = [f for f in os.listdir(out_dir) if f.endswith(".csv")]
    if len(produced) != 1:
        sys.exit("expected one label file in {}, found {}".format(out_dir, produced))
    path = os.path.join(out_dir, produced[0])

    gt_path = ev.find_gt(args.gt)
    gt_codes = pm.label_codes(pm.read_label_csv(gt_path))
    result, reason = ev.score_submission(path, gt_codes, tuple(pm.TAUS), args.bootstrap)
    if reason:
        sys.exit("invalid labels: " + reason)

    with open(path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()[:12]
    record = {
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "name": args.name, "note": args.note, "script": args.script, "flags": flags,
        "gt": os.path.basename(gt_path), "commit": git("rev-parse", "--short", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--", "starter", "games.py")),
        "labels_sha": digest, "metrics": summary(result),
    }
    with open(LOG, "a") as handle:
        handle.write(json.dumps(record) + "\n")
    show()


if __name__ == "__main__":
    main()
