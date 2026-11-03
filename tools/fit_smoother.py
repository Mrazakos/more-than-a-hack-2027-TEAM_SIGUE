"""Fit rule 17 (possession/smoother.py) on the DFL open matches, leave one match out (external data only).

    python tools/fit_smoother.py               # LOMO report + final fit -> starter/possession/smoother_dfl.json
    python tools/fit_smoother.py --no-save     # report only

The rules run once per match and degradation seed (context mask, tools/degrade.py) with BASE
(the externally tuned settings, rules 16 and 17 off); their labels and the smoother's features
are kept. Per fold: the logistic regression is fitted on six matches, the transition costs, the
prior strength and rule 16's min_spell are chosen on the same six, and the held-out match is
scored. Rule 16 alone (no smoother), chosen the same way, is the comparison.

The objective is the mean spellF1 over two DFL truths: 'ctrl' (spells under 2 s absorbed, the
challenge's "controlled possession") and 'raw' (DFL's BallPossession as is) -- neither definition
alone decides. Results go to experiments/external_log.jsonl.
"""

import argparse
import dataclasses
import itertools
import json
import os
import sys
import time
import types
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import tune_dfl  # noqa: E402  (puts starter/ and evaluation/ on the path)
import possession_metrics as pm  # noqa: E402
from possession import smoother as sm  # noqa: E402
from possession.context import HalfContext  # noqa: E402
from possession.labeller import label_half  # noqa: E402
from possession.report import Report  # noqa: E402
from possession.rules.kickoff import find_kickoff  # noqa: E402
from possession.rules.postprocess import min_spells  # noqa: E402
from possession.rules.restarts import find_restarts  # noqa: E402
from possession.rules.stoppages import free_kick_stoppages  # noqa: E402
from possession.rules.touch import find_touches  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "starter"))
from externally_tuned import TUNED  # noqa: E402

BASE = dataclasses.replace(TUNED, smoother=False, min_spell=0.0, min_dead=0.0)
SWITCH_TEAM = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
SWITCH_DEAD = (16.0, 24.0, 32.0, 48.0, 64.0, 96.0)
PRIOR = (0.0, 0.25)
MIN_SPELL = (0.0, 0.5, 1.0, 1.5)
MIN_DEAD = 1.0
KEEP_DEAD = ("off", "events", "all")
SUBSAMPLE = 4          # every 4th frame is a training row
CODE = {"H": 0, "A": 1, "D": 2}


CACHE = os.path.join(ROOT, "experiments", "out", "smoother_items")


def collect(data):
    """[(match, seed, halves, gt_raw, gt_ctrl, n)] with per half (first, last, kick, rule labels, X).
    Each item is also pickled to CACHE so the workers read their own instead of being sent it."""
    import pickle
    os.makedirs(CACHE, exist_ok=True)
    items = []
    for match in data.ids:
        for seed in range(data.seeds):
            md = data.match_data(match, BASE, True, seed)
            halves = []
            for half, first, last in md.blocks:
                ctx = HalfContext(md, BASE, half, first, last)
                report = Report()
                labels = label_half(ctx, report)
                kick = find_kickoff(ctx, ctx.tags.kickoffs.get(ctx.half))  # the tables label_half used
                restarts = find_restarts(ctx, kick, find_touches(ctx).gets_it, Report())
                stoppages = free_kick_stoppages(ctx, kick, Report())
                keeps = {mode: sm.kept_dead(ctx, labels, restarts, stoppages, mode) for mode in KEEP_DEAD}
                halves.append((first, last, report.kickoff, labels,
                               sm.features(ctx, labels).astype(np.float32), keeps))
            item = (match, seed, halves, data.raw[match], data.ctrl[match], len(md.ball_x))
            with open(_item_path(len(items)), "wb") as handle:
                pickle.dump(item, handle, protocol=pickle.HIGHEST_PROTOCOL)
            items.append(item)
    return items


def _item_path(index):
    return os.path.join(CACHE, "item_{}.pkl".format(index))


def _load_item(index):
    import pickle
    with open(_item_path(index), "rb") as handle:
        return pickle.load(handle)


def training_rows(items, matches):
    X, y = [], []
    for match, _seed, halves, gt_raw, _c, _n in items:
        if match not in matches:
            continue
        for first, last, kick, _labels, feats, _keeps in halves:
            frames = np.arange(kick, last + 1, SUBSAMPLE)
            X.append(feats[frames - first].astype(float))
            y.append(gt_raw[frames])
    y = np.concatenate(y)
    keep = y <= 2  # H, A, D codes are 0, 1, 2 in possession_metrics
    return np.concatenate(X)[keep], y[keep].astype(int)


def spell_f1(gt, pred):
    return pm.spell_scores(gt, pred, pm.valid_runs(gt))["mean_f1"]


def _fake_ctx(min_spell):
    return types.SimpleNamespace(params=dataclasses.replace(BASE, min_spell=min_spell, min_dead=MIN_DEAD), fps=25)


def score_item(args):
    """Scores of one (match, seed) for every setting: {setting: (ctrl spellF1, raw spellF1)}.
    model None: rule 16 alone."""
    index, model = args
    _match, _seed, halves, gt_raw, gt_ctrl, n = _load_item(index)
    gt_raw_c, gt_ctrl_c = gt_raw, gt_ctrl
    out = {}
    if model is None:
        settings = [(None, None, None, ms, None) for ms in MIN_SPELL]
        variants = {None: [h[3] for h in halves]}
    else:
        settings, variants = [], {}
        grid = list(itertools.product(SWITCH_TEAM, SWITCH_DEAD))
        for prior, mode in itertools.product(PRIOR, KEEP_DEAD):
            per_half = []
            for first, last, kick, labels, feats, keeps in halves:
                start = kick - first
                emission = sm.emissions(model, feats.astype(float), prior, keeps[mode])[start:]
                per_half.append(sm.viterbi_batch(emission, grid))
            for gi, (st, sd) in enumerate(grid):
                halves_labels = []
                for (first, last, kick, labels, _f, _k), paths in zip(halves, per_half):
                    new = labels.copy()
                    new[kick - first:] = np.array(sm.CLASSES, dtype=labels.dtype)[paths[gi]]
                    halves_labels.append(new)
                variants[(st, sd, prior, mode)] = halves_labels
                settings += [(st, sd, prior, ms, mode) for ms in MIN_SPELL]
    for st, sd, prior, ms, mode in settings:
        key = None if model is None else (st, sd, prior, mode)
        full = np.full(n, "X", dtype="<U1")
        for (first, last, _k, _l, _f, _kp), labels in zip(halves, variants[key]):
            lab = labels.copy()
            if ms > 0:
                min_spells(_fake_ctx(ms), lab, Report())
            full[first:last + 1] = lab
        codes = pm.label_codes(full)
        out[(st, sd, prior, ms, mode)] = (spell_f1(gt_ctrl_c, codes), spell_f1(gt_raw_c, codes))
    return out


def table(pool, items, model):
    """{setting: {match: (ctrl, raw) averaged over seeds}}"""
    results = pool.map(score_item, [(i, model) for i in range(len(items))])
    acc = {}
    for item, res in zip(items, results):
        for setting, (c, r) in res.items():
            acc.setdefault(setting, {}).setdefault(item[0], []).append((c, r))
    return {s: {m: tuple(np.mean(v, axis=0)) for m, v in per.items()} for s, per in acc.items()}


def objective(scores, matches):
    return float(np.mean([(scores[m][0] + scores[m][1]) / 2 for m in matches]))


def best_setting(tab, matches):
    return max(tab, key=lambda s: (objective(tab[s], matches), str(s)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    t0 = time.time()
    data = tune_dfl.Data(seeds=args.seeds)
    items = collect(data)
    print("rules run on {} match-seeds ({:.0f}s)".format(len(items), time.time() - t0), flush=True)
    ids = data.ids
    with Pool(args.workers) as pool:
        rules_tab = table(pool, items, None)
        rows = []
        for held in ids:
            train = [m for m in ids if m != held]
            X, y = training_rows(items, train)
            model = sm.fit(X, y)
            tab = table(pool, items, model)
            s17 = best_setting(tab, train)
            s16 = best_setting(rules_tab, train)
            h17, h16 = tab[s17][held], rules_tab[s16][held]
            base = rules_tab[(None, None, None, 0.0, None)][held]
            rows.append((held, base, h16, h17, s16, s17))
            print("hold out {}: rules {:.4f}/{:.4f}  +16 {:.4f}/{:.4f} (min_spell {})  +17 {:.4f}/{:.4f} {}  "
                  "[ctrl/raw]  {:.0f}s".format(held, base[0], base[1], h16[0], h16[1], s16[3], h17[0], h17[1],
                                               s17, time.time() - t0), flush=True)
            tune_dfl.log({"name": "smoother:heldout", "held_out": held, "base": base, "rule16": h16,
                          "rule16_setting": s16[3], "smoother": h17, "smoother_setting": list(s17),
                          "seeds": data.seeds})
        means = {k: np.mean([np.mean(r[i]) for r in rows]) for i, k in ((1, "rules"), (2, "+16"), (3, "+17+16"))}
        ctrl = {k: np.mean([r[i][0] for r in rows]) for i, k in ((1, "rules"), (2, "+16"), (3, "+17+16"))}
        raw = {k: np.mean([r[i][1] for r in rows]) for i, k in ((1, "rules"), (2, "+16"), (3, "+17+16"))}
        print("\nheld-out mean spellF1 (ctrl / raw / mean):")
        for k in means:
            print("  {:8s} {:.4f} / {:.4f} / {:.4f}".format(k, ctrl[k], raw[k], means[k]))
        wins = sum(np.mean(r[3]) > np.mean(r[2]) for r in rows)
        print("  smoother beats rule 16 alone on {} of {} held-out matches".format(wins, len(rows)))

        X, y = training_rows(items, ids)
        model = sm.fit(X, y)
        tab = table(pool, items, model)
        final = best_setting(tab, ids)
    print("final (all 7): switch_team {} switch_dead {} smoother_prior {} min_spell {} keep_dead {} -> objective {:.4f}".format(
        *final, objective(tab[final], ids)))
    model["chosen"] = {"switch_team": final[0], "switch_dead": final[1], "smoother_prior": final[2],
                       "min_spell": final[3], "min_dead": MIN_DEAD, "smoother_keep_dead": final[4]}
    model["heldout"] = {"ctrl": ctrl, "raw": raw}
    model["source"] = ("DFL open data (Bassek et al. 2025, CC BY 4.0), 7 matches, ball hidden by "
                       "tools/degrade.context_mask, base settings starter/externally_tuned.TUNED")
    tune_dfl.log({"name": "smoother:final", "chosen": model["chosen"], "heldout_ctrl": ctrl, "heldout_raw": raw})
    if not args.no_save:
        with open(sm.DEFAULT_MODEL, "w", encoding="utf-8") as handle:
            json.dump(model, handle, indent=1)
        print("saved", os.path.relpath(sm.DEFAULT_MODEL, ROOT))


if __name__ == "__main__":
    main()
