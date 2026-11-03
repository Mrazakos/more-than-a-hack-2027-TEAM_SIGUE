"""Possession labels from the team's own rules -- the best settings so far. No flags needed:

    python starter/best_so_far.py
    python evaluation/run_evaluation.py --bootstrap 1000

writes evaluation/results/best_so_far.csv. Public window (17 possessions in the truth):
spellF1 0.7333, macroF1 0.9340, transF1 0.541, 13 possessions.

The rules, in the order they act (the numbers are the ones used while they were built; 3, 10
and 12 were tried and taken out, and so was a flight rule in 2):

  0. A goalkeeper slot belongs to the team defending the goal on its side of halfway: the ATD's
     goalkeeper slots follow the other team's keeper for minutes at a time. --no-keepers: off.
  1. Before the kickoff the ball is dead. The kickoff is the first frame after Smart Tagging's
     "Kick Off (Start)" tag with the ball seen moving faster than KICK_SPEED (within KICK_SEARCH);
     the team nearest the ball has it. Both halves.
  2. [FLAG] Who has the ball: the team of the nearest player, from the first frame he is within
     TOUCH_RADIUS (1.5 m, --nearest-radius); with nobody that close the last team keeps it.
     Backdating: a change of team is moved back to the frame after the old team last touched the
     ball, at most BACKDATE_MAX (3 s) -- --no-backdate turns it off.
  4. A ball over a goal line or a touchline, by any distance, on one frame, is dead. Over a
     touchline play restarts with the next touch (a throw-in). Over a goal line with a Smart
     Tagging corner within EXIT_TAG_WINDOW: dead until the first kick after the tag, which goes
     to the tagged team.
  5. [FLAG] Otherwise a goal kick: dead until the ball has stood still (PLACED_SPEED) inside a
     penalty area for PLACED_SECONDS and then moves off (KICK_SPEED); it goes to the team
     defending that goal. If neither shows up, play restarts with the next touch.
     --no-goal-kick: always the next touch.
  6. A goal (Smart Tagging's "Kick Off (After Goal)" tags): dead from the last goal-line exit
     before the tag (at most GOAL_LOOKBACK earlier) until the kick after the tag; the team
     nearest the ball then has it.
  7. A free kick from Smart Tagging's "Free Kick" tags: dead from when the ball went quiet before
     the tag (slower than QUIET_SPEED or not seen, at most FK_LOOKBACK earlier) until the first
     kick after the tag, which goes to the tagged team. --no-free-kicks: off.
  8. A stoppage nobody tagged (a foul, an offside, a substitution): dead while the players walk
     -- tempo, the median player speed over 2 s, below SLOW_TEMPO -- and the ball is slower than
     QUIET_SPEED or not seen, for at least SLOW_SECONDS; play restarts with the next touch.
     --no-slow-stop: off.
  8b. Such a stoppage starts where the tempo last stood at SLOW_BACK (1.8 m/s) or more, at most
     SLOW_BACK_MAX earlier -- the players slow down seconds after the whistle -- but never across
     a change of team or a restart. --slow-back 0: off.
  9. Two dead stretches at most MERGE_GAP (5 s) apart, with the players walking between them
     (mean tempo below MERGE_TEMPO), are one stoppage: before a free kick the ball is moved about.
     --merge-gap S sets it, 0 turns it off.
  11. A pass that only goes past an opponent is not his: an opponent within the touch distance
     takes the ball only if its path breaks at him (it turns TURN_DEGREES or its speed changes
     by SPEED_RATIO within PASS_BY_WINDOW frames) or it is slow there (below CONTROL_SPEED).
     --no-pass-by: off.
  13. The stoppages of rules 7 and 8 start DEAD_LEAD (15) frames earlier -- never before the last
     restart or the last change of team. --dead-lead N sets it, 0 turns it off.
  14. A stoppage that restarts with the next touch (throw-in, foul, offside) ends DEAD_TAIL (17)
     frames later -- with 1.5 m play restarts as soon as the taker is within reach -- taking at
     most half of the possession after it. Corners, goal kicks, free-kick tags and goals restart
     at the kick and are not moved. --dead-tail N sets it, 0 turns it off.
  15. The players vote: the team with the ball spreads across the pitch (public window: width
     difference AUC 0.95, starter/team_shape_check.py). A possession of at most SHAPE_MAX (5 s)
     with the other team on both sides, whose team is on average at least SHAPE_VETO (6 m)
     narrower than the other, goes to the other team. --shape-veto M sets it, 0 turns it off.
  17. (off here) A learned smoother: a three-state H/A/D hidden Markov model over the labels so
     far, its per-frame scores from a logistic regression on the rules' labels and tracking
     signals, fitted on the DFL open matches only (possession/smoother.py, tools/fit_smoother.py).
  16. (off here) No possession shorter than MIN_SPELL and no stoppage shorter than MIN_DEAD: the
     shortest such run goes to its neighbours, shortest first (possession/rules/postprocess.py).
     Rules 16 and 17 run last, 17 first; starter/externally_tuned.py --smoother turns both on.

Also: --ball FILE takes the ball from a fused track (starter/ball_fusion.py, from your own ball
detector) instead of the ATD's; --trace FROM TO prints the pass frame by frame (labels unchanged).

Not handled: where the ATD's ball follows the wrong object (review: 11553, 11803, 17354); the end
of an untagged stoppage can still be late. Outfield players keep the team of their ATD slot.

Needs only the repository's games.py, baseline.py and load_smart_tagging.py. The code lives in
starter/possession/ (this file is the entry point: one module per group of rules, see
possession/rules/__init__.py). The first run reads
the ATD once (a minute or two) and caches it in the gitignored data folder; any flag writes
best_so_far_custom.csv (or a name with the flag's setting in it) instead of best_so_far.csv.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import dataclasses  # noqa: E402

from possession.config import Params  # noqa: E402
from possession.cli import main as _main  # noqa: E402
from possession.context import prepare_match  # noqa: E402
from possession.labeller import label_match, possessions  # noqa: E402,F401
from possession.signals import ball_kinematics, path_changes  # noqa: E402,F401  (basic_rules.py, review.py)
from possession.signals import tempo as _tempo  # noqa: E402
from possession.tracking import load_features  # noqa: E402,F401


def tempo(features):
    """Legacy signature used by starter/review.py: takes the features dict."""
    return _tempo(features["player_speed"])


def prepare(data_dir, keepers=True, ball_path=None):
    """Legacy tuple API used by starter/basic_rules.py: (tracking arrays, kickoff tags, corner tags,
    goal tags, team defending the left goal in the first half, free-kick tags)."""
    params = dataclasses.replace(Params(), keepers=keepers)
    return prepare_match(data_dir, params, ball_path).as_legacy()


def main():
    _main(description=__doc__.splitlines()[0])


if __name__ == "__main__":
    main()
