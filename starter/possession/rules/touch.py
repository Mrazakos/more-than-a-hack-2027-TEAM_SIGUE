"""Rules 2 and 11: which frames a player touches the ball, and whether a touch is a real one."""

from typing import NamedTuple

import numpy as np


class Touches(NamedTuple):
    gets_it: np.ndarray  # a player is within touch_radius of the (seen) ball
    played: np.ndarray   # rule 11: the ball's path broke at the player, or it is slow at his feet


def find_touches(ctx):
    p = ctx.params
    with np.errstate(invalid="ignore"):
        gets_it = ctx.seen & (ctx.nearest <= p.touch_radius)

    # Rule 11: did the ball's path break at the player, or is it slow at his feet?
    played = np.ones(len(ctx.x), dtype=bool)
    if p.pass_by:
        broke = ctx.changes.copy()
        for k in range(1, p.pass_by_window + 1):
            broke[k:] |= ctx.changes[:-k]
            broke[:-k] |= ctx.changes[k:]
        with np.errstate(invalid="ignore"):
            played = broke | (ctx.speed < p.control_speed)
    return Touches(gets_it, played)
