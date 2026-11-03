"""Every tuning constant and rule switch of the labeller, in one place.

Params is frozen: the rules read it, nothing writes it. A run with other settings makes a new one
with dataclasses.replace(Params(), dead_lead=0). The defaults are the team's best settings so far.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Params:
    # -- rule switches (the command-line flags) ---------------------------------------------------
    keepers: bool = True           # rule 0 (--no-keepers)
    backdate: bool = True          # rule 2: backdating (--no-backdate)
    pass_by: bool = True           # rule 11 (--no-pass-by)
    goal_kick_rule: bool = True    # rule 5 (--no-goal-kick)
    free_kicks: bool = True        # rule 7 (--no-free-kicks)
    slow_stop: bool = True         # rules 8 and 8b (--no-slow-stop)
    trace: Optional[Tuple[int, int]] = None  # --trace FROM TO

    # -- rule 1: kickoff --------------------------------------------------------------------------
    kick_speed: float = 3.0        # m/s: a ball faster than this has been kicked
    kick_search: int = 250         # frames after a kickoff tag searched for the kick (10 s)
    tag_kickoff_lag: int = 85      # frames, used only if no kick is found: the tag opens ~3.5 s early

    # -- rule 2: who has the ball -----------------------------------------------------------------
    touch_radius: float = 1.5      # m: a player this close to the ball touches it (swept 0.4-2.0)
    backdate_max: int = 75         # frames a change of team may be moved back (3 s)

    # -- rule 11: a pass going by -----------------------------------------------------------------
    turn_degrees: float = 35.0     # a change of direction at least this big is a break in the path...
    speed_ratio: float = 2.0       # ...and so is the speed changing by this factor either way
    min_path_speed: float = 2.0    # m/s: below this the direction is tracking noise, not a path
    path_span: int = 3             # frames before and after a frame used to measure the path
    pass_by_window: int = 3        # frames either side of a touch searched for the break
    control_speed: float = 3.0     # m/s: a ball slower than this at a player is at his feet

    # -- rules 4 and 5: the ball over a goal line -------------------------------------------------
    exit_tag_window: Tuple[int, int] = (-125, 1000)  # frames around the exit in which a corner tag counts
    corner_kick_search: int = 750  # frames after the corner tag searched for the kick (30 s)
    goal_kick_search: int = 1000   # frames after the exit searched for the goal kick (40 s)
    placed_speed: float = 0.5      # m/s, smoothed: a ball this slow in a penalty area has been put down...
    placed_seconds: float = 0.5    # ...for at least this long
    box_depth: float = 16.5        # a penalty area (m)
    box_half_width: float = 20.16

    # -- rule 6: goals ----------------------------------------------------------------------------
    goal_lookback: int = 3000      # frames before an after-goal kickoff tag searched for the goal (2 min)

    # -- rule 7: free kicks -----------------------------------------------------------------------
    quiet_speed: float = 1.5       # m/s, smoothed: a ball slower than this, or not seen, is not in play...
    fk_lookback: int = 375         # ...searched back from a free-kick tag at most this far (15 s)

    # -- rules 8 and 8b: untagged stoppages -------------------------------------------------------
    slow_tempo: float = 1.2        # m/s: median player speed (2 s average) below this: the players walk...
    slow_seconds: float = 1.0      # ...while the ball is quiet, for at least this long
    slow_back: Optional[float] = 1.8  # m/s: rule 8b, the start goes back to where the tempo last stood this high
    slow_back_max: int = 375       # frames, at most (15 s)

    # -- rule 9: merging --------------------------------------------------------------------------
    merge_gap: float = 5.0         # s
    merge_tempo: float = 1.5       # m/s: only a gap the players walk through is joined

    # -- rules 13 and 14: the ends of a stoppage --------------------------------------------------
    dead_lead: int = 15            # frames (swept 5-28: 10-25 equal)
    dead_tail: int = 17            # frames (dead_check: our ends were 17-18 frames early)

    # -- rule 15: team shape ----------------------------------------------------------------------
    shape_veto: float = 6.0        # m (4-7 m all the same on the public window)
    shape_max: float = 5.0         # s: only possessions this short are judged
    shape_smooth: int = 25         # frames the width difference is averaged over

    # -- rule 16: minimum run lengths (0 = off) ---------------------------------------------------
    min_spell: float = 0.0         # s: a shorter possession goes to a neighbouring run
    min_dead: float = 0.0          # s: a shorter stoppage goes to a neighbouring run

    # -- rule 17: learned H/A/D smoother (possession/smoother.py) -------------------------------
    smoother: bool = False         # off by default
    smoother_model: Optional[str] = None  # JSON coefficients; None = possession/smoother_dfl.json
    smoother_prior: float = 1.0    # how far the class shares are divided out (0 = posterior as is)
    switch_team: float = 8.0       # log cost of an H <-> A change
    switch_dead: float = 8.0       # log cost of going into or out of D
    smoother_keep_dead: str = "off"  # dead balls the smoother may not undo: "off", "events" (goal-line
                                     # exits and free-kick tags), "all" (also throw-ins / next touch)

    # -- signal processing ------------------------------------------------------------------------
    smooth_frames: int = 15        # rolling-median window for the smoothed ball speed
    tempo_frames: int = 50         # the tempo: median player speed averaged over this many frames (2 s)
