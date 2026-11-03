"""The team's rule-based possession labeller, as a package.

    config       Params: every tuning constant and rule switch, in one frozen dataclass
    tracking     ATD reader + feature cache (tracking -> per-frame arrays)
    tags         Smart Tagging events (kickoffs, corners, goals, free kicks)
    signals      pure numpy helpers (ball kinematics, path breaks, tempo, runs)
    context      MatchData (everything read once) and HalfContext (what the rules of one half see)
    rules/       one module per group of rules (see rules/__init__.py for the rule -> module map)
    trace        --trace output
    labeller     wires the rules together: label_half / label_match
    cli          argument parsing and output naming

starter/best_so_far.py is the command-line entry point and keeps the authoritative rule list in its
docstring. Rule numbers are historical (3, 10 and 12 were tried and removed): do not renumber.

Importing this package puts the repository root (games.py) and starter/ (baseline.py,
load_smart_tagging.py) on sys.path, as the scripts in starter/ do.
"""

import os
import sys

_STARTER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.dirname(_STARTER)
for _path in (_STARTER, _ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)
