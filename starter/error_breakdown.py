"""Where do a submission's errors come from? Against any ground truth -- the public window or your own.

    python starter/error_breakdown.py --gt data/labels/BAR-ATH_validation_gt.csv
    python starter/error_breakdown.py --gt data/labels/BAR-ATH_validation_gt.csv --submission best_so_far.csv --top 30

Only frames the truth labels (H, A or D) are counted. Bare file names are looked up in
evaluation/results (submissions) and data/labels (truths); without --gt the public window is used.

  1. frames: how many are wrong, and of what kind --
       wrong team        truth H, we say A (or the other way round)
       missed dead       truth D, we say H or A:  late start (just before our D), early end (just
                         after our D), or missed (the whole stoppage, we never say D there)
       false dead        truth H or A, we say D:  early start / late end around a real stoppage,
                         or invented (no real stoppage near it)
  2. dead stretches: how far our start and end are from the truth's, as a median and as a spread
  3. changes of team: real / false / missed, within TOLERANCE frames (as change_diff.py counts them)
  4. possessions: how many of the truth's are found (same team, overlap IoU >= 0.5), how many of
     ours have no partner -- split or merged ones show up here
  5. the longest wrong stretches, with their video time and the review.py command that opens them

Numbers only; needs games.py and the files themselves.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES  # noqa: E402

# Params
SUBMISSION = "best_so_far.csv"
GT_PUBLIC = os.path.join(ROOT, "evaluation", "gt", "{}_possession_gt_public.csv")
TOLERANCE = 50                # frames (2 s): changes of team this close are the same change
NEAR = 250                    # frames (10 s): a wrong D this close to a real stoppage is a boundary error
IOU = 0.5                     # possessions overlapping at least this much are the same possession
TOP = 20
CONTEXT = 250


def read(name, folders):
    for folder in [""] + folders:
        path = os.path.join(folder, name) if folder else name
        if os.path.exists(path):
            return pd.read_csv(path)["label"].astype(str).str.strip().str.upper().values, path
    sys.exit("{} not found (looked in {})".format(name, ", ".join(["."] + folders)))


def runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))  # end exclusive


def segments(labels, graded):
    """[(start, end, label)] maximal runs of one label inside graded frames (end exclusive)."""
    out = []
    for g0, g1 in runs(graded):
        part = labels[g0:g1]
        cuts = np.concatenate(([0], np.flatnonzero(part[1:] != part[:-1]) + 1, [len(part)]))
        out += [(g0 + int(a), g0 + int(b), part[a]) for a, b in zip(cuts[:-1], cuts[1:])]
    return out


def changes(labels, graded):
    out, last = [], None
    for g0, g1 in runs(graded):
        last = None
        for t in range(g0, g1):
            if labels[t] in ("H", "A"):
                if last is not None and labels[t] != last:
                    out.append((t, labels[t]))
                last = labels[t]
    return out


def unmatched(own, other, tol):
    used, out = set(), []
    for t, team in own:
        best = None
        for j, (u, v) in enumerate(other):
            if v == team and abs(u - t) <= tol and j not in used and (
                    best is None or abs(u - t) < abs(other[best][0] - t)):
                best = j
        if best is None:
            out.append((t, team))
        else:
            used.add(best)
    return out


def vtime(frame, fps):
    s = frame // fps
    return "{:d}:{:02d}:{:02d}".format(s // 3600, s % 3600 // 60, s % 60)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gt", default=None, help="ground truth (default: the public window)")
    parser.add_argument("--submission", default=SUBMISSION)
    parser.add_argument("--top", type=int, default=TOP)
    args = parser.parse_args()
    fps = GAMES[GAME]["fps"]

    truth, gt_path = read(args.gt or GT_PUBLIC.format(GAME),
                          [os.path.join(ROOT, "data", "labels"), os.path.join(ROOT, "evaluation", "gt")])
    ours, sub_path = read(args.submission, [os.path.join(ROOT, "evaluation", "results")])
    graded = np.isin(truth, ("H", "A", "D"))
    n = int(graded.sum())
    print("{} against {}: {} graded frames ({:.1f} min)".format(
        os.path.basename(sub_path), os.path.relpath(gt_path, ROOT), n, n / fps / 60.0))

    # 1. frames by kind
    t_live, t_dead = graded & np.isin(truth, ("H", "A")), graded & (truth == "D")
    o_live, o_dead = np.isin(ours, ("H", "A")), ours == "D"
    wrong_team = t_live & o_live & (truth != ours)
    missed_dead = t_dead & ~o_dead
    false_dead = t_live & o_dead
    no_answer = graded & ~np.isin(ours, ("H", "A", "D"))
    # boundary or not
    ours_dead_runs = runs(o_dead)
    truth_dead_runs = runs(t_dead)
    near_ours = np.zeros(len(truth), dtype=bool)
    for a, b in ours_dead_runs:
        near_ours[max(0, a - NEAR):b + NEAR] = True
    near_truth = np.zeros(len(truth), dtype=bool)
    for a, b in truth_dead_runs:
        near_truth[max(0, a - NEAR):b + NEAR] = True
    kinds = []
    md_whole = md_late = md_early = 0
    for a, b in truth_dead_runs:
        inside = o_dead[a:b]
        if not inside.any():
            md_whole += int((~inside).sum())
            continue
        first, last = np.flatnonzero(inside)[[0, -1]]
        md_late += int((~inside[:first]).sum())
        md_early += int((~inside[last + 1:]).sum())
        md_whole += int((~inside[first:last + 1]).sum())  # gaps inside: counted as missed
    fd_boundary = int((false_dead & near_truth).sum())
    fd_invented = int((false_dead & ~near_truth).sum())
    total_wrong = int((wrong_team | missed_dead | false_dead | no_answer).sum())

    def row(name, frames):
        print("   {:<46} {:>7}  {:>5.1f}%  {:>5.1f}% of errors".format(
            name, frames, 100.0 * frames / n, 100.0 * frames / max(1, total_wrong)))
    print("\n1. frames ({} wrong, {:.1f}% of graded)".format(total_wrong, 100.0 * total_wrong / n))
    print("   {:<46} {:>7}  {:>6}  {:>6}".format("kind", "frames", "of all", ""))
    row("wrong team", int(wrong_team.sum()))
    row("missed dead: our stoppage starts late", md_late)
    row("missed dead: our stoppage ends early", md_early)
    row("missed dead: the stoppage missed (or a gap)", md_whole)
    row("false dead: early start / late end", fd_boundary)
    row("false dead: invented (no real stoppage near)", fd_invented)
    if no_answer.any():
        row("no answer (X)", int(no_answer.sum()))

    # 2. dead stretches: start/end offsets
    starts_off, ends_off, found = [], [], 0
    for a, b in truth_dead_runs:
        best, overlap = None, 0
        for c, d in ours_dead_runs:
            ov = min(b, d) - max(a, c)
            if ov > overlap:
                best, overlap = (c, d), ov
        if best:
            found += 1
            starts_off.append(best[0] - a)
            ends_off.append(best[1] - b)
    print("\n2. dead stretches: the truth has {}, we overlap {} of them; we have {}".format(
        len(truth_dead_runs), found, len([1 for a, b in ours_dead_runs if graded[a:b].any()])))
    if starts_off:
        for name, off in (("start", starts_off), ("end", ends_off)):
            q = np.percentile(off, (25, 50, 75))
            print("   {:<5} offset (ours - truth, + = later): median {:+.0f} fr ({:+.1f} s), "
                  "middle half {:+.0f}..{:+.0f} fr".format(name, q[1], q[1] / fps, q[0], q[2]))

    # 3. changes of team
    tc, oc = changes(truth, graded), changes(ours, graded)
    false = unmatched(oc, tc, TOLERANCE)
    missed = unmatched(tc, oc, TOLERANCE)
    print("\n3. changes of team (within {} frames): truth {}, ours {}: {} real, {} false, {} missed".format(
        TOLERANCE, len(tc), len(oc), len(oc) - len(false), len(false), len(missed)))
    if oc and tc:
        p, r = (len(oc) - len(false)) / len(oc), (len(tc) - len(missed)) / len(tc)
        print("   precision {:.2f}  recall {:.2f}".format(p, r))

    # 4. possessions
    ts = [s for s in segments(truth, graded) if s[2] in ("H", "A")]
    os_ = [s for s in segments(ours, graded) if s[2] in ("H", "A")]
    used, matched = set(), 0
    for a, b, k in ts:
        for j, (c, d, m) in enumerate(os_):
            if j in used or m != k or d <= a or c >= b:
                continue
            iou = (min(b, d) - max(a, c)) / float(max(b, d) - min(a, c))
            if iou >= IOU:
                used.add(j)
                matched += 1
                break
    print("\n4. possessions (same team, IoU >= {}): truth {}, ours {}; found {} of the truth's, "
          "{} of ours have no partner".format(IOU, len(ts), len(os_), matched, len(os_) - matched))
    short_t = sum(1 for a, b, k in ts if b - a < 2 * fps)
    print("   truth possessions under 2 s: {} ({:.0f}%)".format(short_t, 100.0 * short_t / max(1, len(ts))))

    # 5. the longest wrong stretches
    wrong = graded & (truth != ours)
    stretch = sorted(runs(wrong), key=lambda r: -(r[1] - r[0]))[:args.top]
    print("\n5. the longest wrong stretches")
    print("   frames            video     len s  truth  ours   review")
    for a, b in stretch:
        tt = pd.Series(truth[a:b]).mode()[0]
        oo = pd.Series(ours[a:b]).mode()[0]
        print("   {:>6}..{:<6}  {}  {:>6.1f}    {}      {}    python starter/review.py --submission {} "
              "--start {} --end {}".format(a, b - 1, vtime(a, fps), (b - a) / float(fps), tt, oo,
                                           os.path.basename(sub_path), max(0, a - CONTEXT), b + CONTEXT))


if __name__ == '__main__':
    main()
