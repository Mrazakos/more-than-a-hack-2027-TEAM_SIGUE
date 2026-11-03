"""Run the team's existing algorithms, untouched, on the DFL matches and score them.

    python tools/run_algorithms_dfl.py

  baseline_*    starter/baseline.py's nearest-player-to-ball team with its majority-vote smoothing
  basic_rules   starter/basic_rules.py at its defaults (Smart Tagging phases off: DFL has none)
  best_so_far   starter/possession (= best_so_far.py) at its default Params

Scored on clean DFL input and on input degraded to the ATD's ball visibility, against the DFL
possession as is ("raw") and with spells under 2 s absorbed ("ctrl", see dfl_data.controlled).
Results are appended to experiments/external_log.jsonl. External data only.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import dfl_data as d  # noqa: E402
import tune_dfl  # noqa: E402
import possession_metrics as pm  # noqa: E402
from possession.config import Params  # noqa: E402
from possession.labeller import label_match  # noqa: E402
import baseline  # noqa: E402
import basic_rules  # noqa: E402


def baseline_labels(md, window):
    seen = ~np.isnan(md.ball_x) & np.isfinite(np.minimum(md.home, md.away))
    near = np.where(md.home <= md.away, baseline.HOME, baseline.AWAY)
    codes = np.where(seen, near, baseline.NOBODY).astype(np.int8)
    labels = baseline.smooth(codes, window)
    for _half, first, last in md.blocks:      # keep the half-time gap unlabelled
        pass
    gap = np.ones(len(labels), dtype=bool)
    for _half, first, last in md.blocks:
        gap[first:last + 1] = False
    labels = labels.astype("<U1")
    labels[gap] = "X"
    return labels


def basic_rules_labels(md):
    """basic_rules reads the pitch, fps and half spans from games.py: point them at this match."""
    game = basic_rules.GAME
    saved_pitch, saved_fps = basic_rules.GAMES[game]["pitch"], basic_rules.GAMES[game]["fps"]
    saved_blocks = basic_rules.half_video_blocks
    basic_rules.GAMES[game]["pitch"], basic_rules.GAMES[game]["fps"] = md.pitch, md.fps
    basic_rules.half_video_blocks = lambda _g: md.blocks
    try:
        labels, _kicks = basic_rules.label_match(md.as_legacy())
    finally:
        basic_rules.GAMES[game]["pitch"], basic_rules.GAMES[game]["fps"] = saved_pitch, saved_fps
        basic_rules.half_video_blocks = saved_blocks
    return labels


def main():
    data = tune_dfl.Data()
    params = Params()
    algorithms = [("baseline_raw", lambda md: baseline_labels(md, 0)),
                  ("baseline_smooth25", lambda md: baseline_labels(md, 25)),
                  ("baseline_smooth50", lambda md: baseline_labels(md, 50)),
                  ("baseline_smooth125", lambda md: baseline_labels(md, 125)),
                  ("basic_rules", basic_rules_labels),
                  ("best_so_far", lambda md: label_match(md, params))]
    print("DFL matches: {}; ATD-like input drops the ball on {:.1f}% of frames\n".format(
        len(data.ids), 100 * data.missing_share))
    print("{:20s} {:9s} {:5s} {:>8s} {:>8s} {:>8s} {:>9s} {:>9s}".format(
        "algorithm", "input", "gt", "spellF1", "macroF1", "transF1", "poss pred", "poss gt"))
    for name, run in algorithms:
        for degraded in (False, True):
            for truth in ("ctrl", "raw"):
                scores = []
                for match in data.ids:
                    md = data.match_data(match, params, degraded)
                    gt = data.ctrl[match] if truth == "ctrl" else data.raw[match]
                    r = pm.evaluate(gt, pm.label_codes(run(md)), bootstrap_n=0)
                    scores.append((r["spell_f1_mean"], r["macro_f1"], r["transitions"][25]["f1"],
                                   r["n_pred_spells"], r["n_gt_spells"]))
                mean = np.mean(scores, axis=0)
                print("{:20s} {:9s} {:5s} {:8.4f} {:8.4f} {:8.4f} {:9.0f} {:9.0f}".format(
                    name, "degraded" if degraded else "clean", truth, *mean))
                tune_dfl.log({"name": "algorithms:" + name, "degraded": degraded, "truth": truth,
                              "matches": list(data.ids),
                              "mean": {"spellF1": mean[0], "macroF1": mean[1], "transF1": mean[2],
                                       "n_pred": mean[3], "n_gt": mean[4]},
                              "per_match_spellF1": [round(s[0], 4) for s in scores]})
        print()


if __name__ == "__main__":
    main()
