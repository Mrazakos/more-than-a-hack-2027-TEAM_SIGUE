"""The pass over the half: the one place that decides, frame by frame, who has the ball.

It combines the event tables (restarts, stoppages, slow stops) with the touches: rule 2 (the team
of the nearest player within touch_radius, with backdating of a change of team), rule 11 (a pass
going by an opponent is not his), and rule 13 (a stoppage starts dead_lead frames earlier).
"""

import numpy as np

from baseline import AWAY, HOME, NOBODY

LABEL = {HOME: "H", AWAY: "A"}


def run_pass(ctx, kick, touches, restarts, stoppages, slow_stops, tracer):
    """Labels H / A / D for video frames ctx.first..ctx.last."""
    p = ctx.params
    first, last = ctx.first, ctx.last
    gets_it, played, near_team = touches.gets_it, touches.played, ctx.near_team
    labels = np.full(ctx.n, "D", dtype="<U1")

    team = near_team[kick] if kick <= last and np.isfinite(ctx.nearest[kick]) else NOBODY
    dead_until, restart_team = kick, team
    floor, last_touch = kick, {HOME: -1, AWAY: -1}
    last_change = -1  # the last frame the ball went from one team to the other
    for t in range(first, last + 1):
        if t in restarts and t >= dead_until:
            dead_until, restart_team, _ = restarts[t]
            team = NOBODY
        if t in stoppages and stoppages[t][0] > dead_until:
            if p.dead_lead and t >= dead_until:  # rule 13
                lead = max(t - p.dead_lead, floor, last_change + 1)
                labels[lead - first:t - first] = "D"
            dead_until, restart_team = stoppages[t]
            team = NOBODY
            labels[t - first] = "D"
        if t in slow_stops and (t >= dead_until or slow_stops[t][0] > dead_until):
            end_s, back = slow_stops[t]
            if t >= dead_until:
                # Rule 8b looking back: from the rule-8b start, unless the ball changed team in
                # between (play was live there), never before the last restart.
                start = max(back if last_change < back else t, floor)
                if p.dead_lead:  # rule 13
                    start = max(start - p.dead_lead, floor, last_change + 1)
                labels[start - first:t - first] = "D"
            dead_until, restart_team = end_s, None  # restarts with the next touch
            team = NOBODY
        if t < dead_until:
            tracer.row(t, team, "dead until {}".format(dead_until))
            continue
        if t == dead_until:
            floor = t  # no change of team is moved back past a restart
            if restart_team not in (None, NOBODY):
                team = restart_team
        why = ""
        if gets_it[t] and not (team not in (NOBODY, near_team[t]) and not played[t]):
            toucher = near_team[t]
            if team not in (NOBODY, toucher):
                last_change = t
                why = "CHANGE to {}".format(LABEL[toucher])
                if p.backdate:
                    start = max(last_touch[team] + 1, t - p.backdate_max, floor)
                    span = labels[start - first:t - first]
                    span[span == LABEL[team]] = LABEL[toucher]
                    if start < t:
                        why += ", backdated to {}".format(start)
            team = toucher
            last_touch[team] = t
        elif gets_it[t]:
            why = "opponent touch refused (the ball went by: rule 11)"
        if team != NOBODY:
            labels[t - first] = LABEL[team]
        tracer.row(t, team, why)
    return labels
