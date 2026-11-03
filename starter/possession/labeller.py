"""Wiring: the stages of one half, then the whole match.

    1. event tables   kickoff -> touches -> restarts -> free-kick and slow stoppages
    2. the pass       rules/pass_loop.run_pass  (the only code that writes the labels)
    3. tidy-up        shape veto (15) -> dead tail (14) -> merge gaps (9)
"""

import numpy as np

from games import GAME, half_video_blocks

from .context import HalfContext
from .report import Report
from .rules.kickoff import find_kickoff
from .rules.pass_loop import run_pass
from .rules.postprocess import dead_tail, kick_restart_frames, merge_gaps, shape_veto
from .rules.restarts import find_restarts
from .rules.stoppages import free_kick_stoppages, slow_stoppages
from .rules.touch import find_touches
from .trace import Tracer


def label_half(ctx, report):
    """Labels for one half, video frames ctx.first..ctx.last inclusive."""
    kick = find_kickoff(ctx, ctx.tags.kickoffs.get(ctx.half))
    report.kickoff = kick
    touches = find_touches(ctx)
    restarts = find_restarts(ctx, kick, touches.gets_it, report)
    stoppages = free_kick_stoppages(ctx, kick, report)
    slow_stops = slow_stoppages(ctx, kick, report)
    kick_restarts = kick_restart_frames(kick, restarts, stoppages)

    tracer = Tracer(ctx, touches)
    labels = run_pass(ctx, kick, touches, restarts, stoppages, slow_stops, tracer)

    shape_veto(ctx, labels, report)
    dead_tail(ctx, labels, kick, kick_restarts)
    merge_gaps(ctx, labels, kick, report)
    tracer.flush(labels)
    return labels


def label_match(match, params, reports=None):
    """The full-length label array for one set of settings. reports, if a list, gets one
    (half, kickoff was tagged, Report) per half."""
    labels = np.full(len(match.ball_x), "X", dtype="<U1")
    for half, first, last in half_video_blocks(GAME):
        ctx = HalfContext(match, params, half, first, last)
        report = Report()
        labels[first:last + 1] = label_half(ctx, report)
        if reports is not None:
            reports.append((half, half in match.tags.kickoffs, report))
    return labels


def possessions(labels):
    starts = np.concatenate(([0], np.flatnonzero(labels[1:] != labels[:-1]) + 1))
    return int(np.isin(labels[starts], ("H", "A")).sum())
