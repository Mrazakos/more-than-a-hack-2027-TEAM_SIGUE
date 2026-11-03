"""Command line: flags -> Params -> labels -> evaluation/results/<name>.csv."""

import argparse
import dataclasses
import os
import sys

import numpy as np

from games import GAME, half_video_blocks
from baseline import RESULTS_DIR, write_submission

from . import _ROOT
from .config import Params
from .context import prepare_match
from .labeller import label_match, possessions


def build_parser(description=None):
    d = Params()
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--nearest-radius", type=float, default=d.touch_radius, metavar="M",
                        help="rule 2: the touch distance (default 1.5 m)")
    parser.add_argument("--no-keepers", action="store_true", help="turn off rule 0")
    parser.add_argument("--no-backdate", action="store_true", help="rule 2: no backdating")
    parser.add_argument("--no-goal-kick", action="store_true",
                        help="rule 5 off: a ball over the goal line with no corner restarts with the next touch")
    parser.add_argument("--no-free-kicks", action="store_true", help="turn off rule 7")
    parser.add_argument("--no-slow-stop", action="store_true", help="turn off rules 8 and 8b")
    parser.add_argument("--slow-back", type=float, default=d.slow_back, metavar="M/S",
                        help="rule 8b (default 1.8; 0 = off)")
    parser.add_argument("--merge-gap", type=float, default=d.merge_gap, metavar="S",
                        help="rule 9 (default 5; 0 = off)")
    parser.add_argument("--no-pass-by", action="store_true", help="turn off rule 11")
    parser.add_argument("--dead-lead", type=int, default=d.dead_lead, metavar="N",
                        help="rule 13 (default 15; 0 = off)")
    parser.add_argument("--dead-tail", type=int, default=d.dead_tail, metavar="N",
                        help="rule 14 (default 17; 0 = off)")
    parser.add_argument("--shape-veto", type=float, default=d.shape_veto, metavar="M",
                        help="rule 15 (default 6; 0 = off)")
    parser.add_argument("--shape-max", type=float, default=d.shape_max, metavar="S",
                        help="rule 15: only possessions up to S seconds are judged (default 5)")
    parser.add_argument("--ball", default=None, metavar="FILE",
                        help="a fused ball track from ball_fusion.py instead of the ATD's ball")
    parser.add_argument("--trace", type=int, nargs=2, default=None, metavar=("FROM", "TO"),
                        help="print the pass frame by frame over FROM..TO (labels unchanged)")
    parser.add_argument("--data-dir", default=None, help="data bundle, if not ./data or MTAH_DATA_DIR")
    parser.add_argument("--out", default=RESULTS_DIR, help="where to write")
    return parser


def params_from_args(args):
    return dataclasses.replace(
        Params(),
        touch_radius=args.nearest_radius, keepers=not args.no_keepers,
        backdate=not args.no_backdate, goal_kick_rule=not args.no_goal_kick,
        free_kicks=not args.no_free_kicks, slow_stop=not args.no_slow_stop,
        slow_back=args.slow_back if args.slow_back > 0 else None, merge_gap=args.merge_gap,
        pass_by=not args.no_pass_by, dead_lead=args.dead_lead, dead_tail=args.dead_tail,
        shape_veto=args.shape_veto, shape_max=args.shape_max,
        trace=tuple(args.trace) if args.trace else None)


def output_name(args, argv):
    """The default run is best_so_far; the settings swept most get their own names."""
    named = {"--nearest-radius": "_t{:g}".format(args.nearest_radius),
             "--merge-gap": "_m{:g}".format(args.merge_gap),
             "--slow-back": "_sb{:g}".format(args.slow_back),
             "--dead-lead": "_lead{}".format(args.dead_lead),
             "--dead-tail": "_tail{}".format(args.dead_tail),
             "--shape-veto": "_sv{:g}".format(args.shape_veto),
             "--shape-max": "_smax{:g}".format(args.shape_max),
             "--no-pass-by": "_nopb", "--no-backdate": "_nobd", "--no-free-kicks": "_nofk",
             "--no-goal-kick": "_nogk", "--no-slow-stop": "_noslow", "--no-keepers": "_nok",
             "--ball": "_ball", "--trace": ""}
    flags = [a.split("=")[0] for a in argv if a.startswith("--")]
    name = "best_so_far_custom" if [f for f in flags if f not in named] else "best_so_far"
    return name + "".join(named[f] for f in named if f in flags)


def print_reports(reports, labels):
    blocks = {half: (first, last) for half, first, last in half_video_blocks(GAME)}
    for half, tagged, report in reports:
        part = labels[blocks[half][0]:blocks[half][1] + 1]
        print("half {}: kickoff at {}{}   restarts: {}   possessions {}   dead {:.0f}%".format(
            half, report.kickoff, "" if tagged else " (no tag: half start)",
            ", ".join("{} {}".format(v, k) for k, v in report.counts.items()) or "none",
            possessions(part), 100.0 * np.mean(part == "D")))


def main(description=None, argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = build_parser(description).parse_args(argv)
    params = params_from_args(args)

    reports = []
    match = prepare_match(args.data_dir, params, args.ball)
    labels = label_match(match, params, reports)
    print_reports(reports, labels)

    path = write_submission(os.path.join(args.out, "{}.csv".format(output_name(args, argv))), labels)
    print("\n{}\nscore it:  python evaluation/run_evaluation.py --bootstrap 1000".format(
        os.path.relpath(path, _ROOT)))
