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

Tried and removed (not numbered): Smart Tagging's team-phase tags (progression, build up,
defending...) as a prior on the team. They agree with the truth on only 77% of frames against the
system's 91%, and a veto on short possessions lowered spellF1 0.7333 -> 0.6143
(experiments/log.jsonl, runs pv0.5, pv1.5, pv1_max10).

Also: --ball FILE takes the ball from a fused track (starter/ball_fusion.py, from your own ball
detector) instead of the ATD's; --trace FROM TO prints the pass frame by frame (labels unchanged).

Not handled: where the ATD's ball follows the wrong object (review: 11553, 11803, 17354); the end
of an untagged stoppage can still be late. Outfield players keep the team of their ATD slot.

Needs only the repository's games.py, baseline.py and load_smart_tagging.py. The first run reads
the ATD once (a minute or two) and caches it in the gitignored data folder; any flag writes
best_so_far_custom.csv (or a name with the flag's setting in it) instead of best_so_far.csv.
"""

import argparse
import os
import sys
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from baseline import (AWAY, FIELD_SEP, GROUP_SEP, HOME, MISSING, NOBODY, RESULTS_DIR,  # noqa: E402
                      team_names, team_of_each_track, write_submission)
import load_smart_tagging  # noqa: E402

# Params
# Rule 1: kickoff.
KICK_SPEED = 3.0              # m/s: a ball faster than this has been kicked
KICK_SEARCH = 250             # frames after a kickoff tag searched for the kick (10 s)
TAG_KICKOFF_LAG = 85          # frames, used only if no kick is found: the tag opens ~3.5 s early

# Rule 2: who has the ball.
TOUCH_RADIUS = 1.5            # m: a player this close to the ball touches it (swept 0.4-2.0: 1.4-1.5)
BACKDATE_MAX = 75             # frames a change of team may be moved back (3 s)

# Rule 11: a pass going by.
TURN_DEGREES = 35.0           # a change of direction at least this big is a break in the path...
SPEED_RATIO = 2.0             # ...and so is the speed changing by this factor either way
MIN_PATH_SPEED = 2.0          # m/s: below this the direction is tracking noise, not a path
PATH_SPAN = 3                 # frames before and after a frame used to measure the path
PASS_BY_WINDOW = 3            # frames either side of a touch searched for the break
CONTROL_SPEED = 3.0           # m/s: a ball slower than this at a player is at his feet
PASS_BY = True                # --no-pass-by sets this False

# Rules 4 and 5: the ball over a goal line.
EXIT_TAG_WINDOW = (-125, 1000)  # frames around the exit in which a corner tag means a corner
CORNER_KICK_SEARCH = 750      # frames after the corner tag searched for the kick (30 s)
GOAL_KICK_SEARCH = 1000       # frames after the exit searched for the goal kick (40 s)
PLACED_SPEED = 0.5            # m/s, smoothed: a ball this slow in a penalty area has been put down...
PLACED_SECONDS = 0.5          # ...for at least this long
BOX_DEPTH, BOX_HALF_WIDTH = 16.5, 20.16  # a penalty area
GOAL_KICK_RULE = True         # --no-goal-kick sets this False

# Rule 6: goals.
GOAL_LOOKBACK = 3000          # frames before an after-goal kickoff tag searched for the goal (2 min)

# Rule 7: free kicks.
QUIET_SPEED = 1.5             # m/s, smoothed: a ball slower than this, or not seen, is not in play...
FK_LOOKBACK = 375             # ...searched back from a free-kick tag at most this far (15 s)

# Rules 8 and 8b: untagged stoppages.
SLOW_TEMPO = 1.2              # m/s: median player speed (2 s average) below this: the players walk...
SLOW_SECONDS = 1.0            # ...while the ball is quiet, for at least this long
SLOW_BACK = 1.8               # m/s: rule 8b, the start goes back to where the tempo last stood this high
SLOW_BACK_MAX = 375           # frames, at most (15 s)

# Rule 9: joining dead stretches.
MERGE_GAP = 5.0               # s
MERGE_TEMPO = 1.5             # m/s: only a gap the players walk through is joined

# Rules 13, 14, 15.
DEAD_LEAD = 15                # frames (swept 5-28: 10-25 equal)
DEAD_TAIL = 17                # frames (dead_check: our ends were 17-18 frames early)
SHAPE_VETO = 6.0              # m (4-7 m all the same on the public window)
SHAPE_MAX = 5.0               # s: only possessions this short are judged
SHAPE_SMOOTH = 25             # frames the width difference is averaged over

SMOOTH_FRAMES = 15            # rolling-median window for the smoothed ball speed
TRACE = None                  # --trace FROM TO
LABEL = {HOME: "H", AWAY: "A"}

# Reading the tracking.
MAX_PLAYER_SPEED = 12.0       # m/s; a faster jump between frames is an identity switch
TEMPO_FRAMES = 50             # the tempo: median player speed averaged over this many frames (2 s)
CACHE_NAME = "best_so_far_features_v4_{}.npz"


# -- reading the tracking -------------------------------------------------------------------------

def keeper_sides(meta_path):
    """(goalkeeper slots, team defending the LEFT goal in the first half)."""
    root = ET.parse(meta_path).getroot()
    keepers = set()
    for slot, player in enumerate(root.iter("Player")):
        for parameter in player.iter("ProviderParameter"):
            if (parameter.findtext("Name") == "position_type"
                    and (parameter.findtext("Value") or "").strip().lower() == "goalkeeper"):
                keepers.add(slot)
    home_id = next(root.iter("Team")).get("id")
    for team in root.iter("Team"):
        if team.get("id") != home_id:
            continue
        for parameter in team.iter("ProviderParameter"):
            if parameter.findtext("Name") == "attack_direction_first_half":
                direction = (parameter.findtext("Value") or "").strip().lower()
                return keepers, HOME if direction == "left_to_right" else AWAY
    sys.exit("{}: no attack_direction_first_half for the home team".format(meta_path))


def read_features(raw_path, meta_path, n_frames, pitch, ball=None):
    """Per frame: the ball (m); each team's nearest player to it, by ATD slot (_fixed) and with a
    goalkeeper in the team defending his side (_keeper); the median player speed; each team's
    width (spread across the pitch of its outfield players). ball: a fused track to use instead."""
    length, width = pitch
    halfway = length / 2.0
    teams = team_of_each_track(meta_path)
    keepers, left_first_half = keeper_sides(meta_path)
    blocks = half_video_blocks(GAME)
    second_half_from = (blocks[0][2] + blocks[1][1]) // 2  # middle of halftime

    out = {name: np.full(n_frames, np.nan) for name in
           ("ball_x", "ball_y", "home_fixed", "away_fixed", "home_keeper", "away_keeper",
            "player_speed", "home_width", "away_width")}
    fps = GAMES[GAME]["fps"]
    previous_frame, previous = None, {}

    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1  # frame_id counts from 1, submissions from 0
            if not 0 <= frame < n_frames:
                continue
            positions, speeds = {}, []
            consecutive = previous_frame is not None and frame == previous_frame + 1
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if group[0] == MISSING:
                    continue
                values = group.split(",")
                px, py = float(values[0]) * length, float(values[1]) * width
                positions[slot] = (px, py)
                if consecutive and slot in previous:
                    step = ((px - previous[slot][0]) ** 2 + (py - previous[slot][1]) ** 2) ** 0.5 * fps
                    if step <= MAX_PLAYER_SPEED:
                        speeds.append(step)
            if speeds:
                out["player_speed"][frame] = float(np.median(speeds))
            for team, name in ((HOME, "home_width"), (AWAY, "away_width")):
                ys = [p[1] for s_, p in positions.items() if teams[s_] == team and s_ not in keepers]
                if len(ys) >= 5:
                    out[name][frame] = float(np.std(ys))
            previous_frame, previous = frame, positions
            if ball is not None:
                bx, by = ball[0][frame] * length, ball[1][frame] * width
                if not (np.isfinite(bx) and np.isfinite(by)):
                    continue
            else:
                if parts[2][0] == MISSING:
                    continue
                atd = parts[2].split(",")
                bx, by = float(atd[0]) * length, float(atd[1]) * width
            out["ball_x"][frame], out["ball_y"][frame] = bx, by

            left_team = left_first_half if frame < second_half_from else (
                AWAY if left_first_half == HOME else HOME)
            fixed = [None, None]
            keeper = [None, None]
            for slot, (px, py) in positions.items():
                d = (px - bx) ** 2 + (py - by) ** 2
                team = teams[slot]
                if fixed[team] is None or d < fixed[team]:
                    fixed[team] = d
                if slot in keepers:
                    team = left_team if px < halfway else (AWAY if left_team == HOME else HOME)
                if keeper[team] is None or d < keeper[team]:
                    keeper[team] = d
            for team, name in ((HOME, "home"), (AWAY, "away")):
                if fixed[team] is not None:
                    out[name + "_fixed"][frame] = fixed[team] ** 0.5
                if keeper[team] is not None:
                    out[name + "_keeper"][frame] = keeper[team] ** 0.5
    return out


def load_features(data_dir, ball_path=None):
    """read_features(), cached next to the data so the slow pass runs once (a --ball track has
    its own cache)."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, data_dir)
    name = CACHE_NAME.format(GAME)
    newest = os.path.getmtime(raw_path)
    ball = None
    if ball_path:
        name = name.replace(".npz", "_ball-{}.npz".format(
            os.path.splitext(os.path.basename(ball_path))[0]))
        newest = max(newest, os.path.getmtime(ball_path))
    cache = os.path.join(os.path.dirname(raw_path), name)
    if os.path.isfile(cache) and os.path.getmtime(cache) >= newest:
        with np.load(cache) as stored:
            return {key: stored[key] for key in stored.files}
    n_frames = source_frames("tactical", GAME)
    if ball_path:
        table = pd.read_csv(ball_path)
        ball = (np.full(n_frames, np.nan), np.full(n_frames, np.nan))
        frames = table["frame"].to_numpy()
        ok = (frames >= 0) & (frames < n_frames)
        ball[0][frames[ok]] = table["x"].to_numpy()[ok]
        ball[1][frames[ok]] = table["y"].to_numpy()[ok]
        print("ball from {}: seen on {:.0f}% of frames".format(
            os.path.basename(ball_path), 100.0 * np.isfinite(ball[0]).mean()))
    print("reading {} (once; cached afterwards) ...".format(os.path.basename(raw_path)))
    features = read_features(raw_path, meta_path, n_frames, GAMES[GAME]["pitch"], ball)
    np.savez_compressed(cache, **features)
    return features


def tempo(features):
    """Median player speed, averaged over a centred TEMPO_FRAMES window; NaN where unknown."""
    speed = features["player_speed"]
    known = ~np.isnan(speed)
    total = np.concatenate(([0.0], np.cumsum(np.where(known, speed, 0.0))))
    count = np.concatenate(([0], np.cumsum(known)))
    half = TEMPO_FRAMES // 2
    index = np.arange(len(speed))
    low, high = np.maximum(0, index - half), np.minimum(len(speed), index + half)
    n = count[high] - count[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n >= half, (total[high] - total[low]) / n, np.nan)


# -- helpers --------------------------------------------------------------------------------------

def ball_kinematics(x, y, fps):
    """(speed, smoothed speed) in m/s, NaN where the ball is not seen on both ends."""
    k = 2
    speed = np.full(len(x), np.nan)
    speed[k:-k] = np.hypot(x[2 * k:] - x[:-2 * k], y[2 * k:] - y[:-2 * k]) * fps / (2.0 * k)
    smooth = pd.Series(speed).rolling(SMOOTH_FRAMES, center=True,
                                      min_periods=SMOOTH_FRAMES // 2).median().to_numpy()
    return speed, smooth


def path_changes(x, y, fps):
    """True where the ball's path turns or changes speed (rule 11), from PATH_SPAN frames each side."""
    k = PATH_SPAN
    n = len(x)
    vx_in = np.full(n, np.nan); vy_in = np.full(n, np.nan)
    vx_out = np.full(n, np.nan); vy_out = np.full(n, np.nan)
    vx_in[k:] = x[k:] - x[:-k]; vy_in[k:] = y[k:] - y[:-k]
    vx_out[:-k] = x[k:] - x[:-k]; vy_out[:-k] = y[k:] - y[:-k]
    s_in = np.hypot(vx_in, vy_in) * fps / k
    s_out = np.hypot(vx_out, vy_out) * fps / k
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = (vx_in * vx_out + vy_in * vy_out) / (np.hypot(vx_in, vy_in) * np.hypot(vx_out, vy_out))
        turn = (np.degrees(np.arccos(np.clip(cosine, -1, 1))) >= TURN_DEGREES) & (
            np.minimum(s_in, s_out) >= MIN_PATH_SPEED)
        ratio = np.maximum(s_in, s_out) / np.minimum(s_in, s_out)
        speed_change = (ratio >= SPEED_RATIO) & (np.maximum(s_in, s_out) >= KICK_SPEED)
    return np.nan_to_num(turn, nan=0).astype(bool) | np.nan_to_num(speed_change, nan=0).astype(bool)


def runs_of(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def first_true(mask, start, length):
    hits = np.flatnonzero(mask[start:start + length])
    return start + int(hits[0]) if len(hits) else None


def smart_tagging(data_dir):
    """(kickoff tag frames per half, [(corner tag frame, team)], [after-goal kickoff tag frames],
    [(free-kick tag frame, team)])."""
    meta_path = require(["atd_meta"], GAME, data_dir)[0]
    home, away = (name.strip().lower() for name in team_names(meta_path))
    teams = {home: HOME, away: AWAY}
    kickoffs, corners, goals, free_kicks = {}, [], [], []
    for event in load_smart_tagging.load(GAME, data_dir):
        kind = (event["tags"].get("Type") or "").strip().lower()
        if kind == "kick off (start)":
            half = 1 if event["tags"].get("Half") == "1st Half" else 2
            kickoffs[half] = event["start_frame"]
        elif kind == "kick off (after goal)":
            goals.append(event["start_frame"])
        elif kind in ("corner kick", "free kick"):
            (corners if kind == "corner kick" else free_kicks).append(
                (event["start_frame"],
                 teams.get((event["tags"].get("Team") or "").strip().lower(), NOBODY)))
    return kickoffs, sorted(corners), sorted(goals), sorted(free_kicks)


# -- the rules ------------------------------------------------------------------------------------

def label_half(first, last, f, kick_tag, corners, left_team, report, touch_radius=TOUCH_RADIUS,
               goals=(), backdate=True, free_kicks=(), slow_stop=True, merge_gap=MERGE_GAP):
    """Labels for one half, video frames first..last inclusive."""
    length, width = GAMES[GAME]["pitch"]
    fps = GAMES[GAME]["fps"]
    x, y = f["ball_x"], f["ball_y"]
    home, away = f["home"], f["away"]
    nearest = np.minimum(home, away)
    near_team = np.where(home <= away, HOME, AWAY)
    seen = ~np.isnan(x)
    speed, smooth = f["speed"], f["smooth"]
    n = last - first + 1
    labels = np.full(n, "D", dtype="<U1")

    # Rule 1: the kickoff.
    kick = None
    if kick_tag is not None:
        kick = first_true(seen & (speed > KICK_SPEED), max(kick_tag, first), KICK_SEARCH)
        if kick is None:
            kick = kick_tag + TAG_KICKOFF_LAG
    kick = max(kick if kick is not None else first, first)
    report["kickoff"] = kick

    # Rule 2 (FLAG): a player within touch_radius touches the ball.
    with np.errstate(invalid="ignore"):
        gets_it = seen & (nearest <= touch_radius)

    # Rule 11: did the ball's path break at the player, or is it slow at his feet?
    played = np.ones(len(x), dtype=bool)
    if PASS_BY:
        broke = f["changes"].copy()
        for k in range(1, PASS_BY_WINDOW + 1):
            broke[k:] |= f["changes"][:-k]
            broke[:-k] |= f["changes"][k:]
        with np.errstate(invalid="ignore"):
            played = broke | (speed < CONTROL_SPEED)

    # Rules 4 and 5: exits over a goal line or a touchline, and their restarts.
    with np.errstate(invalid="ignore"):
        over = seen & ((x < 0) | (x > length))
        over_side = seen & ((y < 0) | (y > width)) & ~over
        in_box = seen & (np.minimum(x, length - x) <= BOX_DEPTH) & (
            np.abs(y - width / 2.0) <= BOX_HALF_WIDTH)
        placed_ball = in_box & (smooth < PLACED_SPEED)
    placed_frames = max(1, int(round(PLACED_SECONDS * fps)))
    restarts = {}  # exit frame -> (restart frame, team, kind)
    starts, ends = runs_of(over[first:last + 1])
    for start, end in zip(starts + first, ends + first):
        if start < kick:
            continue
        side_left = x[start] < length / 2.0
        defending = left_team if side_left else (AWAY if left_team == HOME else HOME)
        corner = [(tag, team) for tag, team in corners
                  if start + EXIT_TAG_WINDOW[0] <= tag <= start + EXIT_TAG_WINDOW[1]]
        restart = None
        if corner:
            tag, team = corner[0]
            hit = first_true(seen & (speed > KICK_SPEED), max(tag, end), CORNER_KICK_SEARCH)
            if hit is not None:
                restart = (hit, team if team != NOBODY else None, "corner")
        if restart is None and GOAL_KICK_RULE:  # FLAG(rule 5)
            for p_start, p_end in zip(*runs_of(placed_ball[end:end + GOAL_KICK_SEARCH])):
                if p_end - p_start >= placed_frames:
                    hit = first_true(seen & (speed > KICK_SPEED), end + p_end - SMOOTH_FRAMES // 2,
                                     GOAL_KICK_SEARCH)
                    if hit is not None:
                        restart = (hit, defending, "goal kick")
                    break
        if restart is None:
            hit = first_true(gets_it, end, len(x))
            restart = (hit if hit is not None else last + 1, None, "next touch")
        restarts[start] = restart
        report[restart[2]] = report.get(restart[2], 0) + 1
    goal_line_exits = sorted(restarts)
    starts, ends = runs_of(over_side[first:last + 1])
    for start, end in zip(starts + first, ends + first):
        if start < kick or start in restarts:
            continue
        hit = first_true(gets_it, end, len(x))
        restarts[start] = (hit if hit is not None else last + 1, None, "throw-in")
        report["throw-in"] = report.get("throw-in", 0) + 1

    # Rule 6: goals -- dead from the goal to the kickoff after it.
    for tag in goals:
        if not first <= tag <= last:
            continue
        kick_g = first_true(seen & (speed > KICK_SPEED), tag, KICK_SEARCH)
        kick_g = kick_g if kick_g is not None else tag + TAG_KICKOFF_LAG
        before = [e for e in goal_line_exits if tag - GOAL_LOOKBACK <= e <= tag]
        key = before[-1] if before else tag
        kicker = near_team[kick_g] if kick_g < len(x) and np.isfinite(nearest[kick_g]) else None
        if key in restarts:  # that exit was the goal, not a corner or a goal kick
            old = restarts[key][2]
            report[old] -= 1
            if not report[old]:
                del report[old]
        restarts[key] = (kick_g, kicker, "goal")
        report["goal"] = report.get("goal", 0) + 1

    # Rule 7: free kicks -- dead from the ball going quiet before the tag until the kick after it.
    with np.errstate(invalid="ignore"):
        quiet = ~seen | (smooth < QUIET_SPEED)
    stoppages = {}  # start frame -> (kick frame, team)
    for tag, tag_team in free_kicks:
        if not max(first, kick) <= tag <= last:
            continue
        kick_f = first_true(seen & (speed > KICK_SPEED), tag, CORNER_KICK_SEARCH)
        if kick_f is None:
            continue
        start = tag
        while start > max(first, kick, tag - FK_LOOKBACK) and quiet[start - 1]:
            start -= 1
        stoppages[start] = (kick_f, tag_team if tag_team != NOBODY else None)
        report["free kick"] = report.get("free kick", 0) + 1

    # Rules 8 and 8b: untagged stoppages -- the players walking while the ball is quiet.
    slow_stops = {}  # frame the tempo fell below SLOW_TEMPO -> (first frame after, rule-8b start)
    if slow_stop:
        with np.errstate(invalid="ignore"):
            walking = (f["tempo"] < SLOW_TEMPO) & quiet
        starts, ends = runs_of(walking[first:last + 1])
        for start, end in zip(starts + first, ends + first):
            if start >= kick and end - start >= int(round(SLOW_SECONDS * fps)):
                origin = start
                if SLOW_BACK is not None:
                    floor_ = max(first, kick, start - SLOW_BACK_MAX)
                    while start > floor_ and not f["tempo"][start - 1] >= SLOW_BACK:
                        start -= 1
                slow_stops[origin] = (end, start)
        report["slow stop"] = len(slow_stops)

    # Restarts at a kick, seen to the frame -- rule 14 leaves them alone.
    kick_restarts = {kick}
    kick_restarts.update(r[0] for r in restarts.values() if r[2] in ("corner", "goal kick", "goal"))
    kick_restarts.update(v[0] for v in stoppages.values())

    trace_rows = []

    def trace(t, why):  # --trace
        if TRACE and TRACE[0] <= t <= TRACE[1]:
            side = {HOME: "H", AWAY: "A"}
            cell = lambda v: "{:>6.2f}".format(v) if np.isfinite(v) else "     -"  # noqa: E731
            trace_rows.append((t, side.get(team, "-"), "y" if seen[t] else "-",
                               "{:>5.1f}".format(speed[t]) if np.isfinite(speed[t]) else "    -",
                               side.get(near_team[t], "-") if np.isfinite(nearest[t]) else "-",
                               cell(home[t]), cell(away[t]), "y" if gets_it[t] else "-",
                               "y" if played[t] else "-", why))

    # The pass over the half.
    team = near_team[kick] if kick <= last and np.isfinite(nearest[kick]) else NOBODY
    dead_until, restart_team = kick, team
    floor, last_touch = kick, {HOME: -1, AWAY: -1}
    last_change = -1  # the last frame the ball went from one team to the other
    for t in range(first, last + 1):
        if t in restarts and t >= dead_until:
            dead_until, restart_team, _ = restarts[t]
            team = NOBODY
        if t in stoppages and stoppages[t][0] > dead_until:
            if DEAD_LEAD and t >= dead_until:  # rule 13
                lead = max(t - DEAD_LEAD, floor, last_change + 1)
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
                if DEAD_LEAD:  # rule 13
                    start = max(start - DEAD_LEAD, floor, last_change + 1)
                labels[start - first:t - first] = "D"
            dead_until, restart_team = end_s, None  # restarts with the next touch
            team = NOBODY
        if t < dead_until:
            trace(t, "dead until {}".format(dead_until))
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
                if backdate:
                    start = max(last_touch[team] + 1, t - BACKDATE_MAX, floor)
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
        trace(t, why)

    # Rule 15: a short possession between two of the other team's, whose team stays clearly
    # narrower than the other, is not one.
    if SHAPE_VETO > 0:
        vetoed = 0
        wd = f["width_diff"][first:last + 1]
        bounds = np.concatenate(([0], np.flatnonzero(labels[1:] != labels[:-1]) + 1, [n]))
        for i in range(1, len(bounds) - 2):
            s0, s1 = bounds[i], bounds[i + 1]
            here, before, after = labels[s0], labels[s0 - 1], labels[s1]
            if not (here in ("H", "A") and before == after and before in ("H", "A")
                    and before != here and s1 - s0 <= SHAPE_MAX * fps):
                continue
            d = wd[s0:s1]
            if not np.isfinite(d).any():
                continue
            mine = np.nanmean(d) if here == "H" else -np.nanmean(d)  # its width minus the other's
            if mine <= -SHAPE_VETO:
                labels[s0:s1] = before
                vetoed += 1
        report["shape vetoes"] = vetoed

    # Rule 14: a stoppage that restarts with the next touch ends DEAD_TAIL frames later.
    if DEAD_TAIL > 0:
        before = labels.copy()  # decide on the labels as they were, not as they are being changed
        bounds = np.concatenate(([0], np.flatnonzero(before[1:] != before[:-1]) + 1, [n]))
        for i in range(len(bounds) - 2):
            s0, s1, s2 = bounds[i], bounds[i + 1], bounds[i + 2]
            if before[s0] == "D" and s0 >= kick - first and before[s1] in ("H", "A") and \
                    not any(s0 < r - first <= s1 for r in kick_restarts):
                labels[s1:s1 + min(DEAD_TAIL, (s2 - s1) // 2)] = "D"  # at most half the possession

    # Rule 9: join dead stretches at most merge_gap seconds apart, the players walking between.
    if merge_gap > 0:
        gap_frames, merged = int(round(merge_gap * fps)), 0
        starts, ends = runs_of(labels != "D")
        for start, end in zip(starts, ends):
            if start <= kick - first or end >= n or end - start > gap_frames:
                continue  # not between two dead stretches, or too long
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                walk = np.nanmean(f["tempo"][first + start:first + end])
            if np.isnan(walk) or walk < MERGE_TEMPO:
                labels[start:end] = "D"
                merged += 1
        report["merged gaps"] = merged

    if trace_rows:
        print("  frame  final  holds  seen  speed  near  H dist  A dist  touch  played  what happened")
        for t, holds, sn, sp, nr, hd, ad, tc, pl, why in trace_rows:
            print("  {:>6}    {}      {}     {}   {}    {}   {}  {}    {}      {}   {}".format(
                t, labels[t - first], holds, sn, sp, nr, hd, ad, tc, pl, why))
    return labels


def possessions(labels):
    starts = np.concatenate(([0], np.flatnonzero(labels[1:] != labels[:-1]) + 1))
    return int(np.isin(labels[starts], ("H", "A")).sum())


def prepare(data_dir, keepers=True, ball_path=None):
    """Everything the rules read, once: (tracking arrays, kickoff tags, corner tags, goal tags,
    team defending the left goal in the first half, free-kick tags)."""
    fps = GAMES[GAME]["fps"]
    features = load_features(data_dir, ball_path)
    suffix = "_keeper" if keepers else "_fixed"  # rule 0
    f = {"ball_x": features["ball_x"], "ball_y": features["ball_y"],
         "home": np.nan_to_num(features["home" + suffix], nan=np.inf),
         "away": np.nan_to_num(features["away" + suffix], nan=np.inf)}
    f["speed"], f["smooth"] = ball_kinematics(f["ball_x"], f["ball_y"], fps)
    f["changes"] = path_changes(f["ball_x"], f["ball_y"], fps)  # rule 11
    f["tempo"] = tempo(features)  # rules 8, 9
    f["width_diff"] = pd.Series(features["home_width"] - features["away_width"]).rolling(  # rule 15
        SHAPE_SMOOTH, center=True, min_periods=SHAPE_SMOOTH // 2).mean().to_numpy()
    kickoffs, corners, goals, free_kicks = smart_tagging(data_dir)
    meta_path = require(["atd_meta"], GAME, data_dir)[0]
    _, left_first_half = keeper_sides(meta_path)
    return f, kickoffs, corners, goals, left_first_half, free_kicks


def label_match(prepared, touch_radius=TOUCH_RADIUS, backdate=True, free_kick_rule=True,
                slow_stop=True, merge_gap=MERGE_GAP, reports=None):
    """The full-length label array for one set of settings. reports, if a list, gets one per half."""
    f, kickoffs, corners, goals, left_first_half, free_kicks = prepared
    labels = np.full(len(f["ball_x"]), "X", dtype="<U1")
    for half, first, last in half_video_blocks(GAME):
        left_team = left_first_half if half == 1 else (AWAY if left_first_half == HOME else HOME)
        report = {}
        labels[first:last + 1] = label_half(first, last, f, kickoffs.get(half), corners,
                                            left_team, report, touch_radius, goals, backdate,
                                            free_kicks if free_kick_rule else (), slow_stop,
                                            merge_gap)
        if reports is not None:
            reports.append((half, half in kickoffs, report))
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nearest-radius", type=float, default=TOUCH_RADIUS, metavar="M",
                        help="rule 2: the touch distance (default 1.5 m)")
    parser.add_argument("--no-keepers", action="store_true", help="turn off rule 0")
    parser.add_argument("--no-backdate", action="store_true", help="rule 2: no backdating")
    parser.add_argument("--no-goal-kick", action="store_true",
                        help="rule 5 off: a ball over the goal line with no corner restarts with the next touch")
    parser.add_argument("--no-free-kicks", action="store_true", help="turn off rule 7")
    parser.add_argument("--no-slow-stop", action="store_true", help="turn off rules 8 and 8b")
    parser.add_argument("--slow-back", type=float, default=SLOW_BACK, metavar="M/S",
                        help="rule 8b (default 1.8; 0 = off)")
    parser.add_argument("--merge-gap", type=float, default=MERGE_GAP, metavar="S",
                        help="rule 9 (default 5; 0 = off)")
    parser.add_argument("--no-pass-by", action="store_true", help="turn off rule 11")
    parser.add_argument("--dead-lead", type=int, default=DEAD_LEAD, metavar="N",
                        help="rule 13 (default 15; 0 = off)")
    parser.add_argument("--dead-tail", type=int, default=DEAD_TAIL, metavar="N",
                        help="rule 14 (default 17; 0 = off)")
    parser.add_argument("--shape-veto", type=float, default=SHAPE_VETO, metavar="M",
                        help="rule 15 (default 6; 0 = off)")
    parser.add_argument("--shape-max", type=float, default=SHAPE_MAX, metavar="S",
                        help="rule 15: only possessions up to S seconds are judged (default 5)")
    parser.add_argument("--ball", default=None, metavar="FILE",
                        help="a fused ball track from ball_fusion.py instead of the ATD's ball")
    parser.add_argument("--trace", type=int, nargs=2, default=None, metavar=("FROM", "TO"),
                        help="print the pass frame by frame over FROM..TO (labels unchanged)")
    parser.add_argument("--data-dir", default=None, help="data bundle, if not ./data or MTAH_DATA_DIR")
    parser.add_argument("--out", default=RESULTS_DIR, help="where to write")
    args = parser.parse_args()
    globals().update(SLOW_BACK=args.slow_back if args.slow_back > 0 else None,
                     GOAL_KICK_RULE=not args.no_goal_kick, PASS_BY=not args.no_pass_by,
                     DEAD_LEAD=args.dead_lead, DEAD_TAIL=args.dead_tail,
                     SHAPE_VETO=args.shape_veto, SHAPE_MAX=args.shape_max, TRACE=args.trace)

    reports = []
    labels = label_match(prepare(args.data_dir, not args.no_keepers, args.ball), args.nearest_radius,
                         not args.no_backdate, not args.no_free_kicks, not args.no_slow_stop,
                         args.merge_gap, reports=reports)
    blocks = {half: (first, last) for half, first, last in half_video_blocks(GAME)}
    for half, tagged, report in reports:
        part = labels[blocks[half][0]:blocks[half][1] + 1]
        kickoff = report.pop("kickoff")
        print("half {}: kickoff at {}{}   restarts: {}   possessions {}   dead {:.0f}%".format(
            half, kickoff, "" if tagged else " (no tag: half start)",
            ", ".join("{} {}".format(v, k) for k, v in report.items()) or "none",
            possessions(part), 100.0 * np.mean(part == "D")))

    # the default run is best_so_far.csv; the settings swept most get their own names
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
    flags = [a.split("=")[0] for a in sys.argv[1:] if a.startswith("--")]
    name = "best_so_far_custom" if [f for f in flags if f not in named] else "best_so_far"
    name += "".join(named[f] for f in named if f in flags)
    path = write_submission(os.path.join(args.out, "{}.csv".format(name)), labels)
    print("\n{}\nscore it:  python evaluation/run_evaluation.py --bootstrap 1000".format(
        os.path.relpath(path, ROOT)))



if __name__ == '__main__':
    main()
