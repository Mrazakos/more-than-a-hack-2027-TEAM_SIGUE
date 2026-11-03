"""Unit tests for the pure parts of starter/possession (no data bundle needed).

    python -m unittest discover tests -v
"""

import dataclasses
import os
import sys
import types
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "starter"))

from possession.config import Params  # noqa: E402
from possession.report import Report  # noqa: E402
from possession.rules import postprocess  # noqa: E402
from possession.signals import (first_true, path_changes, runs_of, segment_bounds,  # noqa: E402
                                tempo)


def labels_of(text):
    return np.array(list(text), dtype="<U1")


def fake_ctx(params=None, fps=25, first=0, n=0, tempo_=None, width_diff=None):
    return types.SimpleNamespace(params=params or Params(), fps=fps, first=first,
                                 tempo=tempo_, width_diff=width_diff)


class Signals(unittest.TestCase):
    def test_runs_of(self):
        starts, ends = runs_of(np.array([0, 1, 1, 0, 1, 0, 1, 1], dtype=bool))
        self.assertEqual(list(starts), [1, 4, 6])
        self.assertEqual(list(ends), [3, 5, 8])

    def test_first_true(self):
        mask = np.array([0, 0, 1, 0, 1], dtype=bool)
        self.assertEqual(first_true(mask, 0, 5), 2)
        self.assertEqual(first_true(mask, 3, 5), 4)
        self.assertIsNone(first_true(mask, 0, 2))

    def test_segment_bounds(self):
        self.assertEqual(list(segment_bounds(labels_of("HHDDA"))), [0, 2, 4, 5])

    def test_tempo_needs_half_a_window_of_data(self):
        speed = np.full(200, 2.0)
        speed[:10] = np.nan
        out = tempo(speed)
        self.assertTrue(np.isnan(out[0]))
        self.assertAlmostEqual(out[100], 2.0)

    def test_path_changes_finds_a_turn_not_a_straight_line(self):
        fps = 25
        t = np.arange(40)
        x = np.where(t < 20, t * 0.2, 4.0)           # 5 m/s along x, then along y
        y = np.where(t < 20, 0.0, (t - 20) * 0.2)
        changed = path_changes(x.astype(float), y.astype(float), fps)
        self.assertTrue(changed[17:23].any())
        self.assertFalse(changed[5:10].any())


class LegacyApi(unittest.TestCase):
    def test_best_so_far_keeps_the_names_other_scripts_use(self):
        # basic_rules.py: prepare, path_changes.  review.py: load_features, ball_kinematics, tempo.
        import best_so_far
        for name in ("prepare", "path_changes", "load_features", "ball_kinematics", "tempo"):
            self.assertTrue(callable(getattr(best_so_far, name)), name)

    def test_legacy_tempo_takes_the_features_dict(self):
        import best_so_far
        out = best_so_far.tempo({"player_speed": np.full(200, 2.0)})
        self.assertAlmostEqual(out[100], 2.0)


class Params_(unittest.TestCase):
    def test_frozen(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            Params().dead_lead = 0

    def test_replace(self):
        self.assertEqual(dataclasses.replace(Params(), dead_lead=0).dead_lead, 0)
        self.assertEqual(Params().dead_lead, 15)


class Reporting(unittest.TestCase):
    def test_remove_drops_a_count_that_reaches_zero(self):
        report = Report()
        report.add("corner")
        report.add("goal kick", 2)
        report.remove("corner")
        report.remove("goal kick")
        self.assertEqual(report.counts, {"goal kick": 1})


class Postprocess(unittest.TestCase):
    def test_dead_tail_takes_at_most_half_the_possession(self):
        ctx = fake_ctx(dataclasses.replace(Params(), dead_tail=17))
        labels = labels_of("D" * 5 + "H" * 20)
        out = postprocess.dead_tail(ctx, labels.copy(), kick=0, kick_restarts=set())
        self.assertEqual("".join(out), "D" * 5 + "D" * 10 + "H" * 10)  # (20 // 2) < 17

    def test_dead_tail_spares_restarts_at_a_kick(self):
        ctx = fake_ctx()
        labels = labels_of("D" * 5 + "H" * 40)
        out = postprocess.dead_tail(ctx, labels.copy(), kick=0, kick_restarts={5})
        self.assertEqual("".join(out), "D" * 5 + "H" * 40)

    def test_dead_tail_off(self):
        ctx = fake_ctx(dataclasses.replace(Params(), dead_tail=0))
        labels = labels_of("D" * 5 + "H" * 40)
        self.assertEqual("".join(postprocess.dead_tail(ctx, labels.copy(), 0, set())),
                         "D" * 5 + "H" * 40)

    def test_merge_gaps_joins_a_short_walked_gap(self):
        n = 400
        labels = labels_of("D" * 100 + "H" * 50 + "D" * 250)
        ctx = fake_ctx(tempo_=np.full(n, 0.5))
        report = Report()
        out = postprocess.merge_gaps(ctx, labels, kick=0, report=report)
        self.assertTrue((out == "D").all())
        self.assertEqual(report.counts["merged gaps"], 1)

    def test_merge_gaps_leaves_a_gap_the_players_run_through(self):
        labels = labels_of("D" * 100 + "H" * 50 + "D" * 250)
        ctx = fake_ctx(tempo_=np.full(400, 4.0))
        out = postprocess.merge_gaps(ctx, labels, kick=0, report=Report())
        self.assertEqual((out == "H").sum(), 50)

    def test_shape_veto_gives_a_narrow_blip_back(self):
        # A blip of 'A' between two 'H' stretches, with A 8 m narrower: not a possession.
        labels = labels_of("H" * 30 + "A" * 20 + "H" * 30)
        width_diff = np.full(80, 8.0)  # home width minus away width: away is narrower
        ctx = fake_ctx(width_diff=width_diff)
        ctx.last = 79
        report = Report()
        out = postprocess.shape_veto(ctx, labels, report)
        self.assertTrue((out == "H").all())
        self.assertEqual(report.counts["shape vetoes"], 1)


if __name__ == "__main__":
    unittest.main()
