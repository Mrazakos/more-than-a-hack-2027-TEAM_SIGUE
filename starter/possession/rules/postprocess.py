"""Rules 15, 14 and 9: passes over the finished labels. Each takes labels (in place) and returns it.
They run in this order (labeller.label_half)."""

import warnings

import numpy as np

from ..signals import runs_of, segment_bounds


def shape_veto(ctx, labels, report):
    """Rule 15: a short possession between two of the other team's, whose team stays clearly
    narrower than the other, is not one."""
    p = ctx.params
    if p.shape_veto <= 0:
        return labels
    vetoed = 0
    wd = ctx.width_diff[ctx.first:ctx.last + 1]
    bounds = segment_bounds(labels)
    for i in range(1, len(bounds) - 2):
        s0, s1 = bounds[i], bounds[i + 1]
        here, before, after = labels[s0], labels[s0 - 1], labels[s1]
        if not (here in ("H", "A") and before == after and before in ("H", "A")
                and before != here and s1 - s0 <= p.shape_max * ctx.fps):
            continue
        d = wd[s0:s1]
        if not np.isfinite(d).any():
            continue
        mine = np.nanmean(d) if here == "H" else -np.nanmean(d)  # its width minus the other's
        if mine <= -p.shape_veto:
            labels[s0:s1] = before
            vetoed += 1
    report.set("shape vetoes", vetoed)
    return labels


def kick_restart_frames(kick, restarts, stoppages):
    """Restarts at a kick, seen to the frame -- rule 14 leaves them alone."""
    frames = {kick}
    frames.update(r[0] for r in restarts.values() if r[2] in ("corner", "goal kick", "goal"))
    frames.update(v[0] for v in stoppages.values())
    return frames


def dead_tail(ctx, labels, kick, kick_restarts):
    """Rule 14: a stoppage that restarts with the next touch ends dead_tail frames later, taking
    at most half of the possession after it."""
    p = ctx.params
    if p.dead_tail <= 0:
        return labels
    first = ctx.first
    before = labels.copy()  # decide on the labels as they were, not as they are being changed
    bounds = segment_bounds(before)
    for i in range(len(bounds) - 2):
        s0, s1, s2 = bounds[i], bounds[i + 1], bounds[i + 2]
        if before[s0] == "D" and s0 >= kick - first and before[s1] in ("H", "A") and \
                not any(s0 < r - first <= s1 for r in kick_restarts):
            labels[s1:s1 + min(p.dead_tail, (s2 - s1) // 2)] = "D"  # at most half the possession
    return labels


def merge_gaps(ctx, labels, kick, report):
    """Rule 9: join dead stretches at most merge_gap seconds apart, the players walking between."""
    p = ctx.params
    if p.merge_gap <= 0:
        return labels
    first, n = ctx.first, len(labels)
    gap_frames, merged = int(round(p.merge_gap * ctx.fps)), 0
    starts, ends = runs_of(labels != "D")
    for start, end in zip(starts, ends):
        if start <= kick - first or end >= n or end - start > gap_frames:
            continue  # not between two dead stretches, or too long
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            walk = np.nanmean(ctx.tempo[first + start:first + end])
        if np.isnan(walk) or walk < p.merge_tempo:
            labels[start:end] = "D"
            merged += 1
    report.set("merged gaps", merged)
    return labels
