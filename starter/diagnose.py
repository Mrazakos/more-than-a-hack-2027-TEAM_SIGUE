"""Where and why a possession state machine disagrees with the public ground truth.

Runs possession_v3 (or an earlier one, with --version) from its feature cache, so it takes
seconds, lines its labels up against the five public minutes, and prints three things:

  1. the ground-truth segments, each with the predicted segments that overlap it
  2. every stretch where the two disagree, with the tracking evidence inside it and the state
     machine's own trace -- which rule fired, on which frame -- just before and during it
  3. a tally of the error frames by kind, biggest first

    python starter/diagnose.py
    python starter/diagnose.py --version v2 --min-frames 25 --context 50
    python starter/diagnose.py --submission evaluation/results/baseline_plus_k_h_r2_o_c10.csv

--submission FILE diagnoses any frame,label file instead -- baseline_plus.py's, a baseline's, anyone's.
It has no state machine behind it, so there is no trace; the tracking evidence is still shown.

--probe FRAME prints the raw per-frame signals around one frame instead -- ball position and speed,
distances to each team, team shape around the halfway line, which rules are on -- for asking why
a rule did or did not fire there. It works on any frame of the match, graded or not:

    python starter/diagnose.py --probe 11185 --span 100 --step 5

This is for finding *which rule is wrong*, not for choosing thresholds. Seventeen possessions are
enough to show that a rule misfires and too few to show that one setting beats another -- see
EVALUATION.md.
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, GAMES  # noqa: E402
import possession_v1  # noqa: E402
import possession_v2  # noqa: E402
import possession_v3  # noqa: E402

# Params
GT_PATH = os.path.join(ROOT, "evaluation", "gt", "{}_possession_gt_public.csv")
MIN_FRAMES = 10   # disagreements shorter than this (0.4 s) are timing noise, not a rule misfiring
CONTEXT = 75      # frames of trace shown before each disagreement (3 s)
SPAN = 50         # frames either side of a probed frame
STEP = 1          # print every STEP-th frame of a probe
VERSIONS = {"v1": possession_v1, "v2": possession_v2, "v3": possession_v3}

# Probe columns: (signal, header, format). Signals a version does not compute are skipped.
PROBE_NUMBERS = (("ball_x", "x", "{:>6.1f}"), ("ball_y", "y", "{:>5.1f}"),
                 ("speed", "spd", "{:>5.1f}"), ("smooth_speed", "smth", "{:>5.1f}"),
                 ("d_home", "dH", "{:>5.1f}"), ("d_away", "dA", "{:>5.1f}"),
                 ("centre", "ctr", "{:>5.1f}"), ("violators", "wrong", "{:>5}"),
                 ("n_home", "nH", "{:>3}"), ("n_away", "nA", "{:>3}"),
                 ("tempo", "tempo", "{:>5.2f}"), ("tempo_file", "tfile", "{:>5.1f}"))
PROBE_FLAGS = (("kickoff_setup", "K"), ("kickoff_shape", "K"), ("home_control", "h"),
               ("away_control", "a"), ("still", "s"), ("slow", "w"), ("off_pitch", "o"),
               ("in_box", "b"))


def segments(labels, offset=0):
    """[(first, last, label)] maximal runs, frames inclusive, shifted by offset."""
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(labels)])) - 1
    return [(offset + s, offset + e, labels[s]) for s, e in zip(starts, ends)]


def longest_run(mask):
    """Length of the longest True run."""
    starts, ends = possession_v1.runs(mask)
    return int((ends - starts).max()) if len(starts) else 0


def evidence(signals, first, last):
    """One line of tracking facts about frames first..last inclusive."""
    window = slice(first, last + 1)
    visible = signals["visible"][window]
    with warnings.catch_warnings():
        # A window with no ball is a normal case here, not something to warn about.
        warnings.simplefilter("ignore", RuntimeWarning)
        speed = np.nanmax(signals["speed"][window]) if visible.any() else np.nan
        d_home = np.nanmedian(signals["d_home"][window])
        d_away = np.nanmedian(signals["d_away"][window])
    return ("ball seen {:>3.0f}%  longest gap {:>3} fr  off-pitch {:>3} fr  max speed {:>5.1f} m/s  "
            "median dist H {:>5.1f} m  A {:>5.1f} m  control frames H {:>3}  A {:>3}").format(
        100.0 * visible.mean(), longest_run(~visible), int(signals["off_pitch"][window].sum()),
        speed, d_home, d_away, int(signals["home_control"][window].sum()),
        int(signals["away_control"][window].sum()))


def kind(truth, predicted, first, gt_starts, pred_starts):
    """Name the error: what was confused, and which side moved first."""
    if truth in "HA" and predicted == "D":
        what = "false stoppage"
    elif truth == "D" and predicted in "HA":
        what = "missed stoppage"
    elif truth in "HA" and predicted in "HA":
        what = "wrong team"
    else:
        what = "abstained"
    if first in gt_starts and first not in pred_starts:
        when = "late (truth changed, we did not)"
    elif first in pred_starts and first not in gt_starts:
        when = "early or spurious (we changed, truth did not)"
    else:
        when = "both changed"
    return what, when


def probe(signals, labels, gt, centre, span, step, trace):
    """Per-frame signals for centre-span .. centre+span, one row every `step` frames."""
    numbers = [(key, head, fmt) for key, head, fmt in PROBE_NUMBERS if key in signals]
    flags = [(key, letter) for key, letter in PROBE_FLAGS if key in signals]
    print("\n== probe around frame {}  (flags: {})".format(
        centre, "  ".join("{}={}".format(letter, key) for key, letter in flags)))
    print("  {:>6}  {:>6}  gt pr seen ".format("frame", "dt") + " ".join(
        "{:>{}}".format(head, len(fmt.format(0))) for _, head, fmt in numbers) + "  flags")
    lo, hi = max(0, centre - span), min(len(labels) - 1, centre + span)
    for frame in range(lo, hi + 1, step):
        cells = []
        for key, _, fmt in numbers:
            value = signals[key][frame]
            cells.append(fmt.format(value) if np.isfinite(value) else
                         "{:>{}}".format("-", len(fmt.format(0))))
        on = "".join(letter if signals[key][frame] else "." for key, letter in flags)
        print("  {:>6}  {:>+6.1f}  {}  {}  {}  ".format(
            frame, (frame - centre) / float(GAMES[GAME]["fps"]), gt[frame], labels[frame],
            "y" if signals["visible"][frame] else "n") + " ".join(cells) + "  " + on)
    shown = [entry for entry in trace if lo <= entry[0] <= hi]
    for frame, state, team, why in shown:
        print("  trace {:>6}  {:<4} {}  {}".format(frame, state, team, why or "-"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=None, help="data bundle, if not ./data or MTAH_DATA_DIR")
    parser.add_argument("--min-frames", type=int, default=MIN_FRAMES,
                        help="ignore disagreements shorter than this")
    parser.add_argument("--context", type=int, default=CONTEXT,
                        help="frames of trace to show before each disagreement")
    parser.add_argument("--submission", default=None, metavar="FILE",
                        help="diagnose this frame,label file instead of running a version")
    parser.add_argument("--version", choices=sorted(VERSIONS), default="v3",
                        help="which state machine to diagnose")
    parser.add_argument("--probe", type=int, default=None, metavar="FRAME",
                        help="print the per-frame signals around this video frame and stop")
    parser.add_argument("--ball", default=None, metavar="FILE",
                        help="v3 only: fused ball track from ball_fusion.py")
    parser.add_argument("--gt", default=None, metavar="FILE",
                        help="a frame,label ground truth to compare with, e.g. your own labels "
                             "from label_tool.py --export (default: the public window)")
    parser.add_argument("--keepers-by-side", action="store_true",
                        help="v3 only: goalkeepers take the team defending their side")
    parser.add_argument("--placed-restart", action="store_true",
                        help="v3 only: a restart needs a placed ball")
    parser.add_argument("--set-piece-guard", action="store_true",
                        help="v3 only: set-piece tags move early restarts")
    parser.add_argument("--span", type=int, default=SPAN, help="frames either side of --probe")
    parser.add_argument("--step", type=int, default=STEP, help="print every n-th probed frame")
    args = parser.parse_args()

    fps = GAMES[GAME]["fps"]
    v3_only = args.ball or args.keepers_by_side or args.placed_restart or args.set_piece_guard
    if v3_only and args.version != "v3":
        sys.exit("--ball, --keepers-by-side, --placed-restart and --set-piece-guard need --version v3")
    extra = {}
    if args.ball:
        extra["ball_path"] = args.ball
    if args.keepers_by_side:
        extra["keeper_fix"] = True
    if args.placed_restart:
        extra["placed_restart"] = True
    if args.set_piece_guard:
        extra["set_piece_guard"] = True
    if args.submission:
        path = args.submission
        if not os.path.exists(path):  # a bare name: look in evaluation/results
            path = os.path.join(ROOT, "evaluation", "results", os.path.basename(path))
        labels = pd.read_csv(path)["label"].astype(str).str.strip().str.upper().values
        pitch, fps_ = GAMES[GAME]["pitch"], GAMES[GAME]["fps"]
        signals = possession_v3.frame_signals(
            possession_v3.load_features(GAME, args.data_dir, False), pitch, fps_)
        reports = []
        print("diagnosing {} (no trace: it was not made by a state machine here)".format(
            os.path.basename(args.submission)))
    else:
        labels, signals, reports = VERSIONS[args.version].run(GAME, args.data_dir, **extra)
    trace = [entry for report in reports for entry in report["trace"]]

    gt = pd.read_csv(args.gt or GT_PATH.format(GAME))["label"].astype(str).str.strip().str.upper().values
    if args.probe is not None:
        probe(signals, labels, gt, args.probe, args.span, max(1, args.step), trace)
        return
    graded = np.flatnonzero(gt != "X")
    if not len(graded):
        sys.exit("the ground truth has no graded frames")
    first, last = int(graded[0]), int(graded[-1])
    truth, predicted = gt[first:last + 1], labels[first:last + 1]
    gt_segments, pred_segments = segments(truth, first), segments(predicted, first)
    gt_starts = {s for s, _, _ in gt_segments}
    pred_starts = {s for s, _, _ in pred_segments}

    def stamp(frame):
        return "{:>6}  {:>6.1f}s".format(frame, (frame - first) / float(fps))

    print("\n== 1. ground-truth segments, frames {}..{} (times from the first graded frame)".format(
        first, last))
    for s, e, label in gt_segments:
        if label == "X":
            continue  # between labelled windows
        overlap = ["{}{:.1f}s".format(pl, (min(pe, e) - max(ps, s) + 1) / float(fps))
                   for ps, pe, pl in pred_segments if ps <= e and pe >= s]
        print("  {}  {:>5.1f}s  {}   predicted: {}".format(
            stamp(s), (e - s + 1) / float(fps), label, " ".join(overlap)))
    print("  truth {} segments".format(sum(1 for _, _, label in gt_segments if label != "X")))

    print("\n== 2. disagreements of {}+ frames".format(args.min_frames))
    wrong = (truth != predicted) & (truth != "X")
    starts, ends = possession_v1.runs(wrong)
    tally = {}
    for s, e in zip(starts + first, ends + first - 1):
        n = e - s + 1
        # A disagreement can span several (truth, predicted) pairs; split it where either changes.
        for ps, pe, pair in _pairs(gt, labels, s, e):
            what, when = kind(pair[0], pair[1], ps, gt_starts, pred_starts)
            key = (what, when)
            tally[key] = tally.get(key, 0) + (pe - ps + 1)
        if n < args.min_frames:
            continue
        print("\n  {} .. {}  ({:.1f}s)".format(stamp(s), e, n / float(fps)))
        for ps, pe, pair in _pairs(gt, labels, s, e):
            what, when = kind(pair[0], pair[1], ps, gt_starts, pred_starts)
            print("    truth {} / predicted {}  {} fr  -> {}, {}".format(
                pair[0], pair[1], pe - ps + 1, what, when))
        print("    " + evidence(signals, s, e))
        shown = [entry for entry in trace if s - args.context <= entry[0] <= e]
        for frame, state, team, why in shown:
            print("    trace {}  {:<4} {}  {}".format(stamp(frame), state, team, why or "-"))
        if not shown:
            print("    trace  (no state change in or just before this stretch)")

    print("\n== 3. error frames by kind")
    total = int(wrong.sum())
    n_graded = int((truth != "X").sum())
    for (what, when), frames in sorted(tally.items(), key=lambda item: -item[1]):
        print("  {:>5} fr  {:>5.1f}%  {:<16} {}".format(
            frames, 100.0 * frames / max(total, 1), what, when))
    print("  {:>5} fr  of {} graded frames wrong ({:.1f}%)".format(
        total, n_graded, 100.0 * total / n_graded))


def _pairs(gt, labels, first, last):
    """[(first, last, (truth, predicted))] runs of a constant pair, frames inclusive."""
    pairs = np.char.add(gt[first:last + 1], labels[first:last + 1])
    return [(s, e, (p[0], p[1])) for s, e, p in segments(pairs, first)]


if __name__ == '__main__':
    main()
