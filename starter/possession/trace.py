"""--trace FROM TO: print the pass over a frame range, one row per frame."""

import numpy as np

from .rules.pass_loop import LABEL as SIDE


class Tracer:
    def __init__(self, ctx, touches):
        self.ctx, self.touches = ctx, touches
        self.window = ctx.params.trace
        self.rows = []

    def row(self, t, team, why):
        if not (self.window and self.window[0] <= t <= self.window[1]):
            return
        c, tc = self.ctx, self.touches
        cell = lambda v: "{:>6.2f}".format(v) if np.isfinite(v) else "     -"  # noqa: E731
        self.rows.append((t, SIDE.get(team, "-"), "y" if c.seen[t] else "-",
                          "{:>5.1f}".format(c.speed[t]) if np.isfinite(c.speed[t]) else "    -",
                          SIDE.get(c.near_team[t], "-") if np.isfinite(c.nearest[t]) else "-",
                          cell(c.home[t]), cell(c.away[t]), "y" if tc.gets_it[t] else "-",
                          "y" if tc.played[t] else "-", why))

    def flush(self, labels):
        """Print the rows, with the final label of each frame."""
        if not self.rows:
            return
        first = self.ctx.first
        print("  frame  final  holds  seen  speed  near  H dist  A dist  touch  played  what happened")
        for t, holds, sn, sp, nr, hd, ad, tc, pl, why in self.rows:
            print("  {:>6}    {}      {}     {}   {}    {}   {}  {}    {}      {}   {}".format(
                t, labels[t - first], holds, sn, sp, nr, hd, ad, tc, pl, why))
