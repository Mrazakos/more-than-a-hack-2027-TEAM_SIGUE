"""Rules 4, 5 and 6: the ball out of play, and how play restarts.

A restart is (restart frame, team or None, kind). None means the next touch decides the team.
The result of find_restarts is a dict keyed by the frame the ball left the pitch (or, for a goal
with no exit found, the kickoff tag).
"""

import numpy as np

from baseline import NOBODY

from ..signals import first_true, runs_of


def _exit_masks(ctx):
    """(over a goal line, over a touchline but not a goal line): the one definition of an exit."""
    x, y = ctx.x, ctx.y
    with np.errstate(invalid="ignore"):
        over = ctx.seen & ((x < 0) | (x > ctx.length))
        over_side = ctx.seen & ((y < 0) | (y > ctx.width)) & ~over
    return over, over_side


def _goal_line_exits(ctx, kick, gets_it, report):
    """Rules 4 and 5: exits over a goal line -- corner, goal kick, or the next touch."""
    p = ctx.params
    x, y = ctx.x, ctx.y
    over, _ = _exit_masks(ctx)
    with np.errstate(invalid="ignore"):
        in_box = ctx.seen & (np.minimum(x, ctx.length - x) <= p.box_depth) & (
            np.abs(y - ctx.width / 2.0) <= p.box_half_width)
        placed_ball = in_box & (ctx.smooth < p.placed_speed)
    placed_frames = max(1, int(round(p.placed_seconds * ctx.fps)))
    restarts = {}
    starts, ends = runs_of(over[ctx.first:ctx.last + 1])
    for start, end in zip(starts + ctx.first, ends + ctx.first):
        if start < kick:
            continue
        defending = ctx.defending(x[start])
        corner = [(tag, team) for tag, team in ctx.tags.corners
                  if start + p.exit_tag_window[0] <= tag <= start + p.exit_tag_window[1]]
        restart = None
        if corner:
            tag, team = corner[0]
            hit = ctx.first_kick(max(tag, end), p.corner_kick_search)
            if hit is not None:
                restart = (hit, team if team != NOBODY else None, "corner")
        if restart is None and p.goal_kick_rule:  # rule 5
            for p_start, p_end in zip(*runs_of(placed_ball[end:end + p.goal_kick_search])):
                if p_end - p_start >= placed_frames:
                    hit = ctx.first_kick(end + p_end - p.smooth_frames // 2, p.goal_kick_search)
                    if hit is not None:
                        restart = (hit, defending, "goal kick")
                    break
        if restart is None:
            hit = first_true(gets_it, end, len(x))
            restart = (hit if hit is not None else ctx.last + 1, None, "next touch")
        restarts[start] = restart
        report.add(restart[2])
    return restarts


def _throw_ins(ctx, kick, gets_it, restarts, report):
    """Rule 4: over a touchline, play restarts with the next touch."""
    x = ctx.x
    _, over_side = _exit_masks(ctx)
    starts, ends = runs_of(over_side[ctx.first:ctx.last + 1])
    for start, end in zip(starts + ctx.first, ends + ctx.first):
        if start < kick or start in restarts:
            continue
        hit = first_true(gets_it, end, len(x))
        restarts[start] = (hit if hit is not None else ctx.last + 1, None, "throw-in")
        report.add("throw-in")


def _goals(ctx, goal_line_exits, restarts, report):
    """Rule 6: dead from the goal-line exit before the after-goal kickoff tag to the kickoff."""
    p = ctx.params
    x = ctx.x
    for tag in ctx.tags.goals:
        if not ctx.first <= tag <= ctx.last:
            continue
        kick_g = ctx.first_kick(tag, p.kick_search)
        kick_g = kick_g if kick_g is not None else tag + p.tag_kickoff_lag
        before = [e for e in goal_line_exits if tag - p.goal_lookback <= e <= tag]
        key = before[-1] if before else tag
        kicker = (ctx.near_team[kick_g]
                  if kick_g < len(x) and np.isfinite(ctx.nearest[kick_g]) else None)
        if key in restarts:  # that exit was the goal, not a corner or a goal kick
            report.remove(restarts[key][2])
        restarts[key] = (kick_g, kicker, "goal")
        report.add("goal")


def find_restarts(ctx, kick, gets_it, report):
    """exit frame -> (restart frame, team or None, kind), for the whole half."""
    restarts = _goal_line_exits(ctx, kick, gets_it, report)
    goal_line_exits = sorted(restarts)  # before throw-ins and goals are added
    _throw_ins(ctx, kick, gets_it, restarts, report)
    _goals(ctx, goal_line_exits, restarts, report)
    return restarts
