"""Possession from the most basic rules, built up one rule at a time. best_so_far.py stays as it is.

    python starter/basic_rules.py
    python starter/basic_rules.py --radius 1.0

The rules so far:

  1. Kickoff: before the kickoff the ball is dead. The kickoff is the first frame after Smart
     Tagging's "Kick Off (Start)" tag with the ball seen moving faster than KICK_SPEED (within
     KICK_SEARCH); the team of the player nearest the ball has it. Both halves.
  2. Nearest player: a player within RADIUS of the ball (--radius, default 1.6 m) gets it for his
     team, from that frame on. With nobody that close the team that had it keeps it.
  3. Ball out: a ball over a goal line or a touchline -- by any distance, on one frame -- is dead
     until the next touch (rule 2).
  4. Untagged stoppage (a foul, an offside, a substitution): while the players walk -- tempo, the
     median player speed averaged over 2 s, below SLOW_TEMPO -- and the ball is quiet (slower
     than QUIET_SPEED, smoothed, or not seen), for at least SLOW_SECONDS (3 s), the ball is dead;
     play restarts with the next touch. best_so_far.py's rule 8, held 3 s instead of 1 s. First
     half (your labels): 1-6 s tried; 3 s best on macroF1 and transF1 with spellF1 level with
     4-6 s. --no-stoppage turns it off; --stop-seconds S changes the 3 s.
  5. Pass going by: an opponent within RADIUS takes the ball only if its path breaks at him -- it
     turns at least TURN_DEGREES or its speed changes by SPEED_RATIO within PASS_BY_WINDOW frames
     -- or it is slow there (below CONTROL_SPEED, at his feet). A ball that goes straight on at
     speed passed him by. Touches by the team that has the ball, and the first touch after a
     dead ball, are not affected. best_so_far.py's rule 11. --no-pass-by turns it off.
  6. Corner: a ball over a goal line with a Smart Tagging "Corner Kick" tag within EXIT_TAG_WINDOW
     of the exit is dead until the first kick after the tag (faster than KICK_SPEED, within
     CORNER_KICK_SEARCH), and then the tagged team has it. --no-corners turns it off.
  7. Goal kick: a ball over a goal line with no corner tag is dead until it has stood still
     (smoothed speed below PLACED_SPEED) inside a penalty area for PLACED_SECONDS and is then
     kicked (faster than KICK_SPEED); then the team defending that goal has it. --no-goal-kicks
     turns it off. Without either (or if neither shows up) the ball restarts with the next touch.
  8. Joining stoppages: two dead stretches at most MERGE_GAP (3 s) apart, with the players walking
     in between (mean tempo below MERGE_TEMPO), are one stoppage: before a free kick the ball is
     set down, rolled and passed to the taker, and each move would end the dead ball for a
     moment. --merge-gap S sets the gap, 0 turns it off.
  9. Earlier start of an untagged stoppage: a rule-4 stoppage starts where the tempo last stood at
     SLOW_BACK (1.8 m/s) or more, at most SLOW_BACK_MAX (15 s) earlier -- the players slow down
     seconds after the whistle -- but never across a change of team (play was live there) and
     never before the last restart. --slow-back V sets the tempo, 0 turns it off.
  10. Width veto: the team with the ball spreads across the pitch. A possession of at most
     SHAPE_MAX (5 s) with the other team on both sides, whose team is on average at least
     SHAPE_VETO (4 m) narrower than the other while it lasts, goes to the other team. Uses the
     players only, not the ball. Applied before rule 8. --shape-veto M sets it, 0 turns it off.
  11. Smart Tagging phases (both on, "all", by default). On your first half the tagged team has the ball
     in 95% of a BUILD UP and 78% of an ATTACKING TRANSITION (starter/phase_check.py) -- more
     than our labels there (86%, 63%). --build-up live / all: inside a BUILD UP, our live frames
     (live) or our live and dead frames (all) go to the tagged team; --transition live / all:
     the same inside an ATTACKING TRANSITION. Applied before rule 8. First half: both "all"
     spellF1 0.4622 -> 0.4843, macroF1 0.8124 -> 0.8246. "off" turns either off.

Every run writes evaluation/results/basic_rules.csv (or basic_rules_r<radius>.csv) and scores it
straight away with the official metrics: on the public window, and on your own labels if
data/labels/BAR-ATH_validation_gt.csv is there (--gt to choose another file). Your labels are the
first half: tune on it, and keep the second half for one check at the end.

The tracking is read with best_so_far.py's reader and cache (data only, none of its rules).
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from games import GAME, GAMES, half_video_blocks  # noqa: E402
from baseline import AWAY, HOME, NOBODY, RESULTS_DIR, write_submission  # noqa: E402
import possession_metrics as pm  # noqa: E402
import load_smart_tagging  # noqa: E402
from games import require  # noqa: E402
from baseline import team_names  # noqa: E402
import best_so_far  # noqa: E402  -- only for reading the tracking and the kickoff tags

# Params
# Rule 1: kickoff.
KICK_SPEED = 3.0              # m/s: a ball faster than this has been kicked
KICK_SEARCH = 250             # frames after the kickoff tag searched for the kick (10 s)
TAG_KICKOFF_LAG = 85          # frames, if no kick is found: the tag opens ~3.5 s early
# Rule 2: nearest player.
RADIUS = 1.6                  # m (first half, with rule 5: 1.6 best; 0.5 was best without it)
# Rule 4: untagged stoppages.
SLOW_TEMPO = 1.2              # m/s: the players walk below this...
QUIET_SPEED = 1.5             # m/s, smoothed: ...while the ball is slower than this, or not seen...
SLOW_SECONDS = 3.0            # ...for at least this long
# Rules 6 and 7: corners and goal kicks.
EXIT_TAG_WINDOW = (-125, 1000)  # frames around the exit in which a corner tag means a corner
CORNER_KICK_SEARCH = 750      # frames after the corner tag searched for the kick (30 s)
GOAL_KICK_SEARCH = 1000       # frames after the exit searched for the goal kick (40 s)
PLACED_SPEED = 0.5            # m/s, smoothed: a ball this slow in a penalty area has been put down...
PLACED_SECONDS = 0.5          # ...for at least this long
BOX_DEPTH, BOX_HALF_WIDTH = 16.5, 20.16  # a penalty area
# Rule 9: earlier start of an untagged stoppage.
SLOW_BACK = 1.8               # m/s: back to where the tempo last stood this high...
SLOW_BACK_MAX = 375           # ...at most this many frames earlier (15 s)
# Rule 10: width veto.
SHAPE_VETO = 4.0              # m (first half: 3-5 m best, 4 m spellF1 0.4622; 2 m swallows real ones)
SHAPE_MAX = 5.0               # s: only possessions this short are judged
# Rule 8: joining stoppages.
MERGE_GAP = 3.0               # s (first half: 3-12 s all within 0.002; 3 s nearest the possession count)
MERGE_TEMPO = 1.5             # m/s: only a gap the players walk through is joined
# Rule 5: a pass going by (the break in the ball's path comes from best_so_far.path_changes).
PASS_BY_WINDOW = 3            # frames either side of a touch searched for the break
CONTROL_SPEED = 3.0           # m/s: a ball slower than this at a player is at his feet

OUTPUT_NAME = "basic_rules"
LABEL = {HOME: "H", AWAY: "A"}
GT_PUBLIC = os.path.join(ROOT, "evaluation", "gt", "{}_possession_gt_public.csv")
GT_MINE = os.path.join(ROOT, "data", "labels", "{}_validation_gt.csv")


def phase_tags(data_dir, code):
    """[(start frame, end frame, team)] of the Smart Tagging phases with this code."""
    meta_path = require(["atd_meta"], GAME, data_dir)[0]
    home, away = (name.strip().lower() for name in team_names(meta_path))
    teams = {home: HOME, away: AWAY}
    out = []
    for e in load_smart_tagging.load(GAME, data_dir):
        team = teams.get((e["tags"].get("Team") or "").strip().lower())
        if (e["code"] or "").strip().upper() == code and team is not None and e["end_frame"] > e["start_frame"]:
            out.append((e["start_frame"], e["end_frame"], team))
    return out


def first_true(mask, start, length):
    hits = np.flatnonzero(mask[start:start + length])
    return start + int(hits[0]) if len(hits) else None


def label_half(first, last, f, kick_tag, radius, stoppage=True, stop_seconds=SLOW_SECONDS,
               pass_by=True, corners=(), left_team=HOME, corner_rule=True, goal_kick_rule=True,
               merge_gap=MERGE_GAP, slow_back=SLOW_BACK, shape_veto=SHAPE_VETO, phases=()):
    """Labels for one half, video frames first..last inclusive."""
    length, width = GAMES[GAME]["pitch"]
    x, y = f["ball_x"], f["ball_y"]
    home, away = f["home"], f["away"]
    nearest = np.minimum(home, away)
    near_team = np.where(home <= away, HOME, AWAY)
    seen = ~np.isnan(x)
    speed, smooth = f["speed"], f["smooth"]
    fps = GAMES[GAME]["fps"]
    labels = np.full(last - first + 1, "D", dtype="<U1")

    # Rule 1: the kickoff.
    kick = None
    if kick_tag is not None:
        kick = first_true(seen & (speed > KICK_SPEED), max(kick_tag, first), KICK_SEARCH)
        if kick is None:
            kick = kick_tag + TAG_KICKOFF_LAG
    kick = max(kick if kick is not None else first, first)

    # Rule 2: who touches the ball. Rule 3: is it off the pitch?
    with np.errstate(invalid="ignore"):
        touch = seen & (nearest <= radius)
        out = seen & ((x < 0) | (x > length) | (y < 0) | (y > width))

    # Rule 4: the players walking while the ball is quiet, for at least stop_seconds.
    stopped = np.zeros(len(x), dtype=bool)
    stop_starts = {}  # rule 9: first frame of a rule-4 stoppage -> where the tempo last stood high
    if stoppage:
        with np.errstate(invalid="ignore"):
            walking = (f["tempo"] < SLOW_TEMPO) & (~seen | (smooth < QUIET_SPEED))
        edges = np.diff(np.concatenate(([0], walking[first:last + 1].astype(np.int8), [0])))
        for a, b in zip(np.flatnonzero(edges == 1) + first, np.flatnonzero(edges == -1) + first):
            if b - a >= int(round(stop_seconds * fps)):
                stopped[a:b] = True
                back = a
                if slow_back:
                    floor_ = max(first, kick, a - SLOW_BACK_MAX)
                    while back > floor_ and not f["tempo"][back - 1] >= slow_back:
                        back -= 1
                stop_starts[a] = back

    # Rule 5: did the ball's path break at the player, or is it slow at his feet?
    played = np.ones(len(x), dtype=bool)
    if pass_by:
        broke = f["changes"].copy()
        for k in range(1, PASS_BY_WINDOW + 1):
            broke[k:] |= f["changes"][:-k]
            broke[:-k] |= f["changes"][k:]
        with np.errstate(invalid="ignore"):
            played = broke | (speed < CONTROL_SPEED)

    # Rules 6 and 7: a ball over a goal line -- a corner or a goal kick, and when it is taken.
    with np.errstate(invalid="ignore"):
        over_goal_line = seen & ((x < 0) | (x > length))
        in_box = seen & (np.minimum(x, length - x) <= BOX_DEPTH) & (
            np.abs(y - width / 2.0) <= BOX_HALF_WIDTH)
        placed = in_box & (smooth < PLACED_SPEED)
        kicked = seen & (speed > KICK_SPEED)
    placed_frames = max(1, int(round(PLACED_SECONDS * fps)))
    restarts = {}  # exit frame -> (restart frame, team)
    edges = np.diff(np.concatenate(([0], over_goal_line[first:last + 1].astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(edges == 1) + first, np.flatnonzero(edges == -1) + first):
        if a < kick:
            continue
        restart = None
        tags = [(tag, tm) for tag, tm in corners
                if a + EXIT_TAG_WINDOW[0] <= tag <= a + EXIT_TAG_WINDOW[1]]
        if corner_rule and tags:
            tag, tm = tags[0]
            hit = first_true(kicked, max(tag, b), CORNER_KICK_SEARCH)
            if hit is not None:
                restart = (hit, tm if tm != NOBODY else NOBODY)
        if restart is None and goal_kick_rule and not tags:
            defending = left_team if x[a] < length / 2.0 else (AWAY if left_team == HOME else HOME)
            p_edges = np.diff(np.concatenate(([0], placed[b:b + GOAL_KICK_SEARCH].astype(np.int8), [0])))
            for p0, p1 in zip(np.flatnonzero(p_edges == 1), np.flatnonzero(p_edges == -1)):
                if p1 - p0 >= placed_frames:
                    hit = first_true(kicked, b + p1 - 7, GOAL_KICK_SEARCH)  # 7: half the smoothing window
                    if hit is not None:
                        restart = (hit, defending)
                    break
        if restart is not None:
            restarts[a] = restart

    team = near_team[kick] if np.isfinite(nearest[kick]) else NOBODY
    dead = False
    dead_until, restart_team = None, NOBODY
    last_change, floor = -1, kick  # rule 9: the last change of team, and the last restart
    for t in range(kick, last + 1):
        if t in stop_starts and not dead and stop_starts[t] < t:
            back = stop_starts[t]
            start = max(back if last_change < back else t, floor)  # rule 9
            labels[start - first:t - first] = "D"
        if t in restarts and dead_until is None:  # rules 6, 7: dead until the corner / goal kick
            dead_until, restart_team = restarts[t]
            dead, team = True, NOBODY
        if dead_until is not None:
            if t < dead_until:
                continue
            dead, team, dead_until = False, restart_team, None
            floor = t
            if team != NOBODY:
                labels[t - first] = LABEL[team]
                continue
        if out[t] or stopped[t]:        # rules 3 and 4: dead from the frame it is out or stopped...
            dead, team = True, NOBODY
        elif touch[t] and (team in (NOBODY, near_team[t]) or played[t]):
            if dead:
                floor = t                       # a restart
            elif team not in (NOBODY, near_team[t]):
                last_change = t                 # a change of team
            dead, team = False, near_team[t]  # ...until the next touch (rule 2); rule 5 for opponents
        if not dead and team != NOBODY:
            labels[t - first] = LABEL[team]

    # Rule 10: a short possession between two of the other team's, whose team stays clearly
    # narrower than the other, is not one.
    if shape_veto > 0:
        wd = f["width_diff"][first:last + 1]  # Barcelona's width minus Athletic's, 1 s average
        cuts = np.concatenate(([0], np.flatnonzero(labels[1:] != labels[:-1]) + 1, [len(labels)]))
        for i in range(1, len(cuts) - 2):
            s0, s1 = cuts[i], cuts[i + 1]
            here, before, after = labels[s0], labels[s0 - 1], labels[s1]
            if not (here in ("H", "A") and before == after and before in ("H", "A")
                    and before != here and s1 - s0 <= SHAPE_MAX * fps):
                continue
            d = wd[s0:s1]
            d = d[np.isfinite(d)]
            if len(d) and (d.mean() if here == "H" else -d.mean()) <= -shape_veto:
                labels[s0:s1] = before

    # Rule 11: inside a trusted Smart Tagging phase, the tagged team has the ball.
    for tags, mode in phases:
        for a, b, tm in tags:
            a, b = max(a, kick), min(b, last + 1)
            if a >= b:
                continue
            span = labels[a - first:b - first]
            change = np.isin(span, ("H", "A")) if mode == "live" else np.ones(len(span), dtype=bool)
            span[change] = LABEL[tm]

    # Rule 8: a short live gap between two dead stretches, the players walking, is dead too.
    if merge_gap > 0:
        live = labels != "D"
        edges = np.diff(np.concatenate(([0], live.astype(np.int8), [0])))
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if a <= kick - first or b >= len(labels) or b - a > merge_gap * fps:
                continue  # not between two dead stretches, or too long
            walk = f["tempo"][first + a:first + b]
            walk = walk[np.isfinite(walk)]
            if not len(walk) or walk.mean() < MERGE_TEMPO:
                labels[a:b] = "D"
    return labels, kick


def label_match(prepared, radius=RADIUS, stoppage=True, stop_seconds=SLOW_SECONDS, pass_by=True,
                corner_rule=True, goal_kick_rule=True, merge_gap=MERGE_GAP, slow_back=SLOW_BACK,
                shape_veto=SHAPE_VETO, phases=()):
    f, kickoffs, corners, _, left_first_half = prepared[:5]
    labels = np.full(len(f["ball_x"]), "X", dtype="<U1")
    kicks = {}
    for half, first, last in half_video_blocks(GAME):
        left_team = left_first_half if half == 1 else (AWAY if left_first_half == HOME else HOME)
        labels[first:last + 1], kicks[half] = label_half(
            first, last, f, kickoffs.get(half), radius, stoppage, stop_seconds, pass_by,
            corners, left_team, corner_rule, goal_kick_rule, merge_gap, slow_back, shape_veto,
            phases)
    return labels, kicks


def score(labels, gt_path):
    gt = pm.read_label_csv(gt_path)
    r = pm.evaluate(pm.label_codes(gt), pm.label_codes(labels), bootstrap_n=200)
    return r


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--radius", type=float, default=RADIUS, metavar="M",
                        help="rule 2: the touch distance (default 1.6 m)")
    parser.add_argument("--no-stoppage", action="store_true", help="turn off rule 4")
    parser.add_argument("--stop-seconds", type=float, default=SLOW_SECONDS, metavar="S",
                        help="rule 4: how long the players must walk (default 3 s)")
    parser.add_argument("--no-pass-by", action="store_true", help="turn off rule 5")
    parser.add_argument("--no-corners", action="store_true", help="turn off rule 6")
    parser.add_argument("--no-goal-kicks", action="store_true", help="turn off rule 7")
    parser.add_argument("--slow-back", type=float, default=SLOW_BACK, metavar="M/S",
                        help="rule 9: a stoppage starts where the tempo was last this high (default 1.8; 0 = off)")
    parser.add_argument("--shape-veto", type=float, default=SHAPE_VETO, metavar="M",
                        help="rule 10: metres narrower (default 4; 0 = off)")
    parser.add_argument("--build-up", choices=("off", "live", "all"), default="all",
                        help="rule 11: inside a BUILD UP the tagged team has the ball (default all)")
    parser.add_argument("--transition", choices=("off", "live", "all"), default="all",
                        help="rule 11: inside an ATTACKING TRANSITION the tagged team has the ball "
                             "(default all)")
    parser.add_argument("--merge-gap", type=float, default=MERGE_GAP, metavar="S",
                        help="rule 8: join stoppages at most S seconds apart (default 3; 0 = off)")
    parser.add_argument("--gt", default=None, help="your labels (default: data/labels/..._validation_gt.csv)")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()

    prepared = best_so_far.prepare(args.data_dir)
    phases = []
    if args.build_up != "off":
        phases.append((phase_tags(args.data_dir, "BUILD UP"), args.build_up))
    if args.transition != "off":
        phases.append((phase_tags(args.data_dir, "ATTACKING TRANSITION"), args.transition))
    labels, kicks = label_match(prepared, args.radius, not args.no_stoppage, args.stop_seconds,
                                not args.no_pass_by, not args.no_corners, not args.no_goal_kicks,
                                args.merge_gap, args.slow_back, args.shape_veto, phases)
    name = OUTPUT_NAME + ("" if args.radius == RADIUS else "_r{:g}".format(args.radius))
    name += "_nostop" if args.no_stoppage else (
        "" if args.stop_seconds == SLOW_SECONDS else "_ss{:g}".format(args.stop_seconds))
    name += "_nopb" if args.no_pass_by else ""
    name += "_nocorner" if args.no_corners else ""
    name += "_nogk" if args.no_goal_kicks else ""
    name += "" if args.merge_gap == MERGE_GAP else "_m{:g}".format(args.merge_gap)
    name += "" if args.slow_back == SLOW_BACK else "_sb{:g}".format(args.slow_back)
    name += "" if args.shape_veto == SHAPE_VETO else "_sv{:g}".format(args.shape_veto)
    name += "" if args.build_up == "all" else "_bu-" + args.build_up
    name += "" if args.transition == "all" else "_tr-" + args.transition
    path = write_submission(os.path.join(RESULTS_DIR, "{}.csv".format(name)), labels)
    print("kickoffs at {}; wrote {}".format(kicks, os.path.relpath(path, ROOT)))

    mine = args.gt or GT_MINE.format(GAME)
    print("\n  scored on               spellF1  (5-95%)          macroF1  transF1  poss (truth)  dead")
    for title, gt_path in (("public window", GT_PUBLIC.format(GAME)), ("your labels", mine)):
        if not os.path.exists(gt_path):
            print("  {:<22}  (no file: {})".format(title, os.path.relpath(gt_path, ROOT)))
            continue
        r = score(labels, gt_path)
        graded = np.isin(pm.read_label_csv(gt_path)["label"].astype(str).str.upper().values,
                         ("H", "A", "D"))
        print("  {:<22}  {:.4f}   ({:.3f}..{:.3f})   {:.4f}   {:.4f}   {:>4} ({:>4})  {:>3.0f}%".format(
            title, r["spell_f1_mean"], r["spell_bootstrap_low"], r["spell_bootstrap_high"],
            r["macro_f1"], r["transitions"][25]["f1"], r["n_pred_spells"], r["n_gt_spells"],
            100.0 * np.mean(labels[graded] == "D")))


if __name__ == '__main__':
    main()
