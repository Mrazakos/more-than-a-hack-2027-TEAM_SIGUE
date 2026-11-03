"""Score and tune the possession labeller on the DFL open matches (external data only).

    python tools/tune_dfl.py --baseline            # the current Params on all 7 matches
    python tools/tune_dfl.py --tune                # leave-one-match-out coordinate descent
    python tools/tune_dfl.py --tune --pass 2       # more passes over the parameter groups

Nothing here reads BAR-ATH labels or video; the BAR-ATH tracking is read only for the ball's
run-length statistics (tools/degrade.py). Every evaluation is appended to
experiments/external_log.jsonl (never experiments/log.jsonl).

How a value is chosen. For each held-out match the search runs on the other six: coordinate
descent over PARAM_GRID, a value is taken only if it beats the incumbent by more than MIN_GAIN
on the training mean of spellF1 (ties and noise keep the default). The held-out match is then
scored with that fold's parameters and with the defaults: if tuning does not beat the defaults on
held-out matches, it is not worth adopting. The final proposal is the same search on all seven.
"""

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import dfl_data as d  # noqa: E402
import degrade  # noqa: E402
import possession_metrics as pm  # noqa: E402
from possession.config import Params  # noqa: E402
from possession.labeller import label_match  # noqa: E402
from possession.signals import tempo  # noqa: E402

LOG = os.path.join(ROOT, "experiments", "external_log.jsonl")
OUT_DIR = os.path.join(ROOT, "experiments", "out")
MIN_SECONDS = 2.0     # the 'controlled' ground truth (see dfl_data.controlled)
MIN_GAIN = 0.004      # spellF1 a change must add on the training mean to be taken

# Coarse grids around the defaults, grouped by what they decide. Switches are 0/1.
PARAM_GRID = [
    ("touch_radius", [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]),
    ("backdate_max", [0, 25, 50, 75, 100]),
    ("pass_by", [False, True]),
    ("control_speed", [2.0, 3.0, 4.0]),
    ("turn_degrees", [25.0, 35.0, 50.0]),
    ("speed_ratio", [1.5, 2.0, 3.0]),
    ("slow_stop", [False, True]),
    ("slow_tempo", [0.8, 1.0, 1.2, 1.5]),
    ("slow_seconds", [1.0, 2.0, 3.0]),
    ("quiet_speed", [1.0, 1.5, 2.0]),
    ("merge_gap", [0.0, 2.0, 3.0, 5.0, 8.0]),
    ("dead_lead", [0, 10, 15, 25]),
    ("dead_tail", [0, 10, 17, 25]),
    ("shape_veto", [0.0, 4.0, 6.0, 8.0]),
    ("min_spell", [0.0, 1.0, 2.0, 3.0, 4.0]),
    ("min_dead", [0.0, 0.5, 1.0, 2.0, 3.0]),
]


class Data:
    """The matches, parsed once, with the two ground truths and the degraded ball masks."""

    def __init__(self, ids=None, seeds=2):
        """The ball is hidden by degrade.context_mask (the ATD's missing rate by tempo and pitch
        zone); degraded scores average over `seeds` masks."""
        self.ids = ids or d.MATCH_IDS
        self.seeds = seeds
        table = degrade.atd_context_table()
        self.matches, self.raw, self.ctrl, self.masks = {}, {}, {}, {}
        shares = []
        for i, match in enumerate(self.ids):
            m = d.parse_match(match)
            self.matches[match] = m
            raw = d.ground_truth(m)
            self.raw[match] = pm.label_codes(raw)
            self.ctrl[match] = pm.label_codes(d.controlled(raw, MIN_SECONDS))
            blocks = [tuple(int(v) for v in b) for b in m["blocks"]]
            match_tempo = tempo(d._player_speed(m), Params())
            masks = [degrade.context_mask(blocks, match_tempo, m["ball_x"], float(m["pitch"][0]),
                                          table, seed=1000 + 100 * s + i) for s in range(seeds)]
            self.masks[match] = masks
            inside = np.concatenate([np.arange(a, b + 1) for _h, a, b in blocks])
            shares += [float(np.mean(~mk[inside])) for mk in masks]
        self.missing_share = float(np.mean(shares))
        self._cache = {}

    def match_data(self, match, params, degraded, seed=0):
        key = (match, degraded, seed if degraded else -1, params.smooth_frames, params.tempo_frames,
               params.shape_smooth, params.turn_degrees, params.speed_ratio, params.min_path_speed,
               params.path_span)
        if key not in self._cache:
            mask = self.masks[match][seed] if degraded else None
            self._cache[key] = d.build_match(self.matches[match], match, params, ball_mask=mask)
        return self._cache[key]

    def score(self, match, params, degraded=True, truth="ctrl"):
        """Scores for one match; on degraded input, the mean over the degradation seeds.
        truth: 'ctrl', 'raw', or 'both' (the mean of the two)."""
        gts = {"ctrl": [self.ctrl[match]], "raw": [self.raw[match]],
               "both": [self.ctrl[match], self.raw[match]]}[truth]
        rows = []
        for seed in (range(self.seeds) if degraded else [0]):
            md = self.match_data(match, params, degraded, seed)
            pred = pm.label_codes(label_match(md, params))
            for gt in gts:
                r = pm.evaluate(gt, pred, bootstrap_n=0)
                rows.append({"spellF1": r["spell_f1_mean"], "macroF1": r["macro_f1"],
                             "transF1": r["transitions"][25]["f1"] if 25 in r["transitions"] else float("nan"),
                             "n_pred": r["n_pred_spells"], "n_gt": r["n_gt_spells"],
                             "team_acc": r["team_acc_both_alive"], "alive_dead_f1": r["alive_dead_f1"]})
        return {k: float(np.mean([row[k] for row in rows])) for k in rows[0]}


# -- parallel scoring: one Data per worker process, the matches spread over the workers ----------
_WORKER_DATA = None


def _init_worker(seeds):
    global _WORKER_DATA
    _WORKER_DATA = Data(seeds=seeds)


def _score_task(args):
    match, params, degraded, truth = args
    return _WORKER_DATA.score(match, params, degraded, truth)


def make_pool(data, workers):
    """A process pool whose workers each hold their own Data (same masks: the seeds are fixed)."""
    from multiprocessing import Pool
    return Pool(workers, initializer=_init_worker, initargs=(data.seeds,))


def commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True).stdout.strip()
    except OSError:
        return ""


def log(entry):
    entry = dict(entry, time=time.strftime("%Y-%m-%dT%H:%M:%S"), dataset="dfl", commit=commit(),
                 gt_min_seconds=MIN_SECONDS)
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


TRUTH = "ctrl"  # the truth descend() optimises; main() --truth
START = None    # the Params the search starts from (None: Params()); main() --start
POOL = None  # set by main() with --workers; mean_over then spreads the matches over it


def mean_over(data, ids, params, degraded=True, truth=None, name="eval", note=""):
    truth = truth or TRUTH
    if POOL is not None:
        results = POOL.map(_score_task, [(m, params, degraded, truth) for m in ids])
        scores = dict(zip(ids, results))
    else:
        scores = {m: data.score(m, params, degraded, truth) for m in ids}
    avg = {k: float(np.nanmean([s[k] for s in scores.values()])) for k in ("spellF1", "macroF1", "transF1")}
    log({"name": name, "note": note, "degraded": degraded, "truth": truth, "matches": list(ids),
         "seeds": data.seeds if degraded else None,
         "params_changed": changed(params), "mean": avg,
         "per_match": {m: {k: round(v, 4) for k, v in s.items()} for m, s in scores.items()}})
    return avg, scores


def changed(params):
    base = Params()
    return {f.name: getattr(params, f.name) for f in dataclasses.fields(Params)
            if getattr(params, f.name) != getattr(base, f.name)}


def descend(data, train_ids, start, passes, tag):
    """Coordinate descent over PARAM_GRID on train_ids (degraded input, controlled truth)."""
    best = start
    best_avg, _ = mean_over(data, train_ids, best, name=tag + ":start")
    for p in range(passes):
        moved = False
        for field, grid in PARAM_GRID:
            incumbent = getattr(best, field)
            cand_best, cand_val = best_avg["spellF1"], incumbent
            for value in grid:
                if value == incumbent:
                    continue
                trial = dataclasses.replace(best, **{field: value})
                avg, _ = mean_over(data, train_ids, trial, name="{}:{}={}".format(tag, field, value))
                if avg["spellF1"] > cand_best + MIN_GAIN:
                    cand_best, cand_val = avg["spellF1"], value
            if cand_val != incumbent:
                best = dataclasses.replace(best, **{field: cand_val})
                best_avg, _ = mean_over(data, train_ids, best, name="{}:take {}={}".format(tag, field, cand_val))
                moved = True
                print("  {} pass {}: {} {} -> {}  (train spellF1 {:.4f})".format(
                    tag, p + 1, field, incumbent, cand_val, best_avg["spellF1"]), flush=True)
        if not moved:
            break
    return best


def baseline(data):
    params = Params()
    print("missing ball share in the degraded input: {:.1f}%".format(100 * data.missing_share))
    for degraded in (False, True):
        for truth in ("raw", "ctrl"):
            avg, scores = mean_over(data, data.ids, params, degraded, truth, name="baseline")
            print("{:9s} gt={:4s}  spellF1 {:.4f}  macroF1 {:.4f}  transF1 {:.4f}   per match spellF1: {}".format(
                "degraded" if degraded else "clean", truth, avg["spellF1"], avg["macroF1"], avg["transF1"],
                " ".join("{:.3f}".format(s["spellF1"]) for s in scores.values())))


def tune(data, passes):
    held_default, held_tuned, folds = [], [], {}
    for held in data.ids:
        train = [m for m in data.ids if m != held]
        print("fold: hold out {}".format(held), flush=True)
        start = START or Params()
        best = descend(data, train, start, passes, "fold-" + held[-4:])
        d0 = data.score(held, start, True, TRUTH)
        d1 = data.score(held, best, True, TRUTH)
        held_default.append(d0["spellF1"]); held_tuned.append(d1["spellF1"])
        folds[held] = {"changed": changed(best), "default": d0["spellF1"], "tuned": d1["spellF1"]}
        log({"name": "heldout", "held_out": held, "params_changed": changed(best),
             "default": d0, "tuned": d1})
        print("   held-out spellF1: default {:.4f} -> tuned {:.4f}".format(d0["spellF1"], d1["spellF1"]), flush=True)
    print("\nheld-out mean spellF1: default {:.4f}, tuned {:.4f}  (gain {:+.4f}, {} of {} folds better)".format(
        np.mean(held_default), np.mean(held_tuned), np.mean(held_tuned) - np.mean(held_default),
        sum(t > dflt for t, dflt in zip(held_tuned, held_default)), len(held_default)))
    print("\nfinal search on all matches")
    final = descend(data, data.ids, START or Params(), passes, "final")
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "dfl_proposal.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump({"changed": changed(final), "folds": folds,
                   "heldout_default": float(np.mean(held_default)),
                   "heldout_tuned": float(np.mean(held_tuned))}, handle, indent=2)
    print("proposal:", changed(final), "->", os.path.relpath(out, ROOT))
    avg, _ = mean_over(data, data.ids, final, name="final-all")
    print("all 7 matches, tuned: spellF1 {:.4f} macroF1 {:.4f} transF1 {:.4f}".format(
        avg["spellF1"], avg["macroF1"], avg["transF1"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--pass", dest="passes", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=2, help="degradation seeds averaged per score")
    parser.add_argument("--truth", choices=("ctrl", "raw", "both"), default="both",
                        help="the DFL truth the search optimises (both: the mean of ctrl and raw)")
    parser.add_argument("--start", choices=("default", "tuned"), default="default",
                        help="start from Params() or from starter/externally_tuned.TUNED")
    parser.add_argument("--workers", type=int, default=7, help="processes (matches scored in parallel)")
    args = parser.parse_args()
    global TRUTH, START, POOL
    TRUTH = args.truth
    if args.start == "tuned":
        sys.path.insert(0, os.path.join(ROOT, "starter"))
        from externally_tuned import TUNED
        START = TUNED
    data = Data(seeds=args.seeds)
    if args.workers > 1:
        POOL = make_pool(data, args.workers)
    if args.baseline or not args.tune:
        baseline(data)
    if args.tune:
        tune(data, args.passes)


if __name__ == "__main__":
    main()
