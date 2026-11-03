"""Rules 7, 8 and 8b: stoppages -- tagged free kicks, and untagged ones found from the players walking."""

import numpy as np

from baseline import NOBODY

from ..signals import runs_of


def free_kick_stoppages(ctx, kick, report):
    """Rule 7: dead from the ball going quiet before a free-kick tag until the first kick after it,
    which goes to the tagged team. Returns {start frame: (kick frame, team or None)}."""
    p = ctx.params
    stoppages = {}
    if not p.free_kicks:
        return stoppages
    for tag, tag_team in ctx.tags.free_kicks:
        if not max(ctx.first, kick) <= tag <= ctx.last:
            continue
        kick_f = ctx.first_kick(tag, p.corner_kick_search)
        if kick_f is None:
            continue
        start = tag
        while start > max(ctx.first, kick, tag - p.fk_lookback) and ctx.quiet[start - 1]:
            start -= 1
        stoppages[start] = (kick_f, tag_team if tag_team != NOBODY else None)
        report.add("free kick")
    return stoppages


def slow_stoppages(ctx, kick, report):
    """Rules 8 and 8b: the players walking (tempo below slow_tempo) while the ball is quiet, for at
    least slow_seconds. Returns {frame the tempo fell: (end of the stretch, rule-8b start)}; play
    restarts with the next touch."""
    p = ctx.params
    slow_stops = {}
    if not p.slow_stop:
        return slow_stops
    with np.errstate(invalid="ignore"):
        walking = (ctx.tempo < p.slow_tempo) & ctx.quiet
    starts, ends = runs_of(walking[ctx.first:ctx.last + 1])
    for start, end in zip(starts + ctx.first, ends + ctx.first):
        if start >= kick and end - start >= int(round(p.slow_seconds * ctx.fps)):
            origin = start
            if p.slow_back is not None:
                floor_ = max(ctx.first, kick, start - p.slow_back_max)
                while start > floor_ and not ctx.tempo[start - 1] >= p.slow_back:
                    start -= 1
            slow_stops[origin] = (end, start)
    report.set("slow stop", len(slow_stops))
    return slow_stops
