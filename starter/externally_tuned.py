"""best_so_far's rules with the settings tuned on the DFL open matches (external data only).

    python starter/externally_tuned.py                  # -> evaluation/results/externally_tuned.csv
    python starter/externally_tuned.py --shape-veto 4   # + the team-shape veto (chosen in 3 of 6 folds)
    python starter/externally_tuned.py --smoother       # + rule 17 (learned smoother) and rule 16

Source: tools/tune_dfl.py, leave-one-match-out coordinate descent over Params on 7 DFL matches
(ball degraded to the ATD's visibility, spells under 2 s absorbed). These six changes were taken
in every one of the six completed folds; the run stopped before fold 7 and the final all-match
search, so the list is the fold consensus, not a recorded proposal. Nothing here was tuned on
BAR-ATH labels. Defaults in possession/config.py are not touched.
"""
import argparse
import dataclasses
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from baseline import RESULTS_DIR, write_submission  # noqa: E402
from possession.config import Params  # noqa: E402
from possession.context import prepare_match  # noqa: E402
from possession.labeller import label_match  # noqa: E402
from possession.cli import print_reports  # noqa: E402

TUNED = dataclasses.replace(
    Params(), touch_radius=1.0, backdate_max=0, speed_ratio=3.0, slow_tempo=0.8,
    dead_lead=0, dead_tail=0)


def with_smoother(params=TUNED, model_path=None):
    """TUNED + rule 17 (learned smoother) + rule 16, with the settings tools/fit_smoother.py chose
    on the DFL matches (stored in the model file under "chosen")."""
    from possession.smoother import load_model
    chosen = load_model(model_path)["chosen"]
    return dataclasses.replace(params, smoother=True, smoother_model=model_path, **chosen)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shape-veto", type=float, default=None, metavar="M",
                        help="also set shape_veto (DFL folds: 4.0 in 3 of 6)")
    parser.add_argument("--smoother", action="store_true",
                        help="add rules 17 and 16 (possession/smoother_dfl.json) -> externally_tuned_smoother.csv")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out", default=RESULTS_DIR)
    args = parser.parse_args()
    params = TUNED if args.shape_veto is None else dataclasses.replace(TUNED, shape_veto=args.shape_veto)
    name = "externally_tuned" + ("" if args.shape_veto is None else "_sv{:g}".format(args.shape_veto))
    if args.smoother:
        params = with_smoother(params)
        name += "_smoother"
    reports = []
    labels = label_match(prepare_match(args.data_dir, params), params, reports)
    print_reports(reports, labels)
    print(os.path.relpath(write_submission(os.path.join(args.out, name + ".csv"), labels)))


if __name__ == "__main__":
    main()
