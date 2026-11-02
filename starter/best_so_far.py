"""Current strongest settings in this repository's rule-based starter. No flags needed:

    python starter/best_so_far.py
    python evaluation/run_evaluation.py --bootstrap 1000

writes evaluation/results/best_so_far.csv.

Public-window score (evaluation/run_evaluation.py): spellF1 0.6207, macroF1 0.8819.

Compared with the earlier 0.5600/0.8698 setup, this version adds: a stricter flight-speed cut,
active-ball guards on stoppage rewinds, tagged goal-kick and throw-in handling, and a short
anti-flicker filter for brief opponent handovers. The same CLI flags remain available; with any
flag the file is written as best_so_far_custom.csv.

Possession from the team's own rules, written from scratch after watching the public window.

The rules, in the order they act:

  0. A goalkeeper slot belongs to the team defending the goal on its side of halfway. The ATD
     gives each team one goalkeeper slot, but starter/keeper_check.py shows them on the wrong half
     for long stretches -- Athletic's slot 46 for ~11 minutes of play (16-230 s at a time),
     Barcelona's slot 24 for ~1 minute -- following the other team's keeper; every touch by that
     keeper then went to the wrong team. --no-keepers turns this off.

  1. Before the kickoff, the ball is dead. The kickoff is the first frame after Smart Tagging's
     "Kick Off (Start)" tag on which the ball is seen moving (the tag opens a few seconds early);
     the team nearest the ball then has it. Applied to both halves.
  2. Who has the ball -- one of two ownership rules, chosen with --ownership:
     nearest (the default)  a touch: the team of the player nearest the ball has it from the
                            first frame he is within TOUCH_RADIUS (0.4 m); with nobody that close
                            -- a ball in the air, or loose -- the last team keeps it
     touch                  the ball changes hands only when its path changes at a player: it
                            turns or changes speed within TOUCH_RADIUS of him, or he keeps a slow
                            ball within TOUCH_RADIUS for CONTROL_FRAMES. A ball in the air
                            passes over players without touching them; a turn with nobody within
                            TOUCH_RADIUS is the ball landing or bouncing, and changes nothing
     From possession_v3, with --no-flight / --no-backdate to turn them off:
     - a ball faster than FLIGHT_SPEED is travelling, not at anyone's feet: nobody touches it;
     - a change of team is moved back to the frame after the old team last touched the ball (at
       most BACKDATE_MAX): the ball changed hands when it left them, not when it arrived.
  3. A goalkeeper's throw or kick: a ball leaving a team's own penalty area faster than
     LAUNCH_SPEED while that team has it stays with that team until the first touch (a player
     within TOUCH_RADIUS) after it left the keeper. --no-launch turns this off.
  4. A ball that leaves the pitch -- over a goal line or a touchline, by any distance, on a single
     frame -- is dead. Over a touchline play restarts with the next touch (a throw-in).
     Over a goal line, it is dead. If a Smart Tagging corner follows
     (within EXIT_TAG_WINDOW), it is a corner: dead until the first kick after the tag, which goes
     to the tagged team.
  5. Otherwise it is a goal kick: dead until the ball has stood still (PLACED_SPEED) inside a
     penalty area for at least --placed-seconds and then moves off (KICK_SPEED); it goes to the
     team defending that goal.
     If neither shows up, play restarts with the next touch (FALLBACK).
  6. A goal, from possession_v3: Smart Tagging's "Kick Off (After Goal)" tags. Dead from the last
     goal-line exit before the tag (at most GOAL_LOOKBACK earlier; else from the tag) until the
     kick, the first frame after the tag with the ball seen moving; the team nearest the ball
     then has it.

  7. A free kick, from Smart Tagging's "Free Kick" tags (the file has only 5 for the match, and no
     offside or foul tags). Dead from when the ball went quiet before the tag -- slower than
     QUIET_SPEED (smoothed) or not seen, at most FK_LOOKBACK earlier -- until the first kick after
     the tag, which goes to the tagged team. --no-free-kicks turns this off.

  8. --slow-stop: a stoppage nobody tagged (a foul, an offside). Dead while the players walk --
     tempo (median player speed, 2 s average) below SLOW_TEMPO -- and the ball is slower than
     QUIET_SPEED or not seen, for at least SLOW_SECONDS; play restarts with the next touch.
     Chosen with starter/stoppage_check.py: in the public window 1.2 m/s held 1 s added 273 of the
     873 dead frames the rules missed, called no live frame dead, and the whole match 41% dead.
     Off by default until the scoreboard says otherwise.

  9. --merge-gap S: two dead stretches at most S seconds apart are one stoppage -- before a free
     kick the ball is moved about (set down, rolled, passed to the taker), and each move would
     otherwise end the dead ball for a moment. The live frames between them become D if the
     players walk there (mean tempo below MERGE_TEMPO; --merge-tempo 99 drops that check).
     --merge-max-ball V: and only if the ball never goes faster than V m/s (smoothed) in the gap
     -- a faster ball is a real restart (a corner, a free kick taken), and play went on after it.

  8a. --restart-cut: an untagged stoppage ends at its restart -- the ball standing still
     (PLACED_SPEED) for RESTART_STILL seconds, then kicked (RESTART_KICK) -- and no new one starts
     within REARM_SECONDS after it or before the players have sped up past SLOW_TEMPO again (at
     most REARM_MAX), nor does rule 9 join across it, nor rule 8b move a start back across it.
  8b. --slow-back V: an untagged stoppage starts when the tempo last stood at V or more (at most
     SLOW_BACK_MAX earlier), not when it has fallen below SLOW_TEMPO -- the players slow down
     seconds after the whistle. Checked on five stoppages timed in the video: with 1.8 the start
     is 0.8-2.7 s off (mean 1.8 s), without it 0.6-24 s late (mean 8.5 s). The video showed stoppages
     running 2-19 s past the restart: the players stand about after a corner or a free kick, the
     ball unseen among them, and that read as walking.

Not handled yet: the end of an untagged stoppage is often late (2-19 s in the video). Outfield players keep the team of their ATD slot.

    python starter/possession_rules.py                          # a touch: nearest player within 0.5 m
    python starter/possession_rules.py --no-keepers             # rule 0 off, for comparison
    python starter/possession_rules.py --ownership touch        # the path-change rule
    python evaluation/run_evaluation.py --bootstrap 1000
    python starter/review.py --submission evaluation/results/possession_rules.csv

The tracking reader and its cache are copied in from baseline_plus.py (data only, none of its
rules), so this file needs only the repository's own games.py, baseline.py and
load_smart_tagging.py. The first run reads the ATD once (a minute or two) and caches it in the
gitignored data folder.
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
TOUCH_RADIUS = 0.4            # m: a player this close to the ball touches it -- everywhere in this file
FLIGHT_SPEED = 11.0           # m/s: a ball faster than this is travelling; nobody touches it
BACKDATE_MAX = 75             # frames a change of team may be moved back (3 s)
TURN_DEGREES = 35.0           # a change of direction at least this big is a change of path...
SPEED_RATIO = 2.0             # ...and so is the speed changing by this factor either way
MIN_PATH_SPEED = 2.0          # m/s: below this the direction is tracking noise, not a path
PATH_SPAN = 3                 # frames before and after a frame used to measure the path
CONTROL_SPEED = 3.0           # m/s: --ownership touch: a ball slower than this, within TOUCH_RADIUS...
CONTROL_FRAMES = 10           # ...for this many frames in a row (0.4 s), is controlled

# Rules 4 and 5: the ball over a goal line.
EXIT_TAG_WINDOW = (-125, 1000)  # frames around the exit in which a corner tag means a corner
CORNER_KICK_SEARCH = 750      # frames after the corner tag searched for the kick (30 s)
GOAL_KICK_SEARCH = 1000       # frames after the exit searched for the goal kick (40 s)
PLACED_SPEED = 0.5            # m/s, smoothed: a ball this slow in a penalty area has been put down...
PLACED_SECONDS = 0.5          # ...for at least this long (--placed-seconds)
BOX_DEPTH, BOX_HALF_WIDTH = 16.5, 20.16  # a penalty area

# Rule 3: a goalkeeper's throw or kick.
LAUNCH_SPEED = 12.0           # m/s: a ball leaving the own penalty area faster than this is launched

# Rule 7: free kicks.
QUIET_SPEED = 1.5             # m/s, smoothed: a ball slower than this, or not seen, is not in play...
ACTIVE_BALL_SPEED = 2.0       # m/s, smoothed: above this, the ball is active for rewind guards
FK_LOOKBACK = 375             # ...searched back from a free-kick tag at most this far (15 s)

# Rule 8: untagged stoppages (--slow-stop).
SLOW_TEMPO = 1.2              # m/s: median player speed (2 s average) below this: the players walk...
SLOW_SECONDS = 1.0            # ...while the ball is quiet, for at least this long

# Rules 8a and 8b (--restart-cut, --slow-back).
RESTART_CUT = False           # set by --restart-cut
RESTART_STILL = 1.0           # s: the ball still this long (smoothed speed below PLACED_SPEED)...
RESTART_KICK = 5.0            # ...then faster than this (m/s) within RESTART_SEARCH: a restart
RESTART_SEARCH = 50           # frames
REARM_SECONDS = 5.0           # no new untagged stoppage this soon after a restart, nor before the
REARM_MAX = 30.0              # players speed up past SLOW_TEMPO again (at most this long, s)
SLOW_BACK = None              # m/s, set by --slow-back
SLOW_BACK_MAX = 375           # frames (15 s)

# Rule 9: joining dead stretches (--merge-gap).
MERGE_TEMPO = 1.5             # m/s: only a gap the players walk through is joined

# Rule 6: goals.
GOAL_LOOKBACK = 3000          # frames before an after-goal kickoff tag searched for the goal (2 min)
SMOOTH_FRAMES = 15            # rolling-median window for the smoothed ball speed
FALLBACK = "next touch"       # when neither a corner nor a goal kick shows up

OUTPUT_NAME = "possession_rules"
LABEL = {HOME: "H", AWAY: "A"}


# -- reading the tracking (copied from baseline_plus.py: data only, none of its rules) ------------

MAX_PLAYER_SPEED = 12.0       # m/s; a faster jump between frames is an identity switch
TEMPO_FRAMES = 50             # the tempo: median player speed averaged over this many frames (2 s)
CACHE_NAME = "baseline_plus_features_v3_{}.npz"  # the same cache file baseline_plus.py wrote


def keeper_sides(meta_path):
    """(goalkeeper slots, team defending the LEFT goal in the first half).

    Goalkeeper slots are the Players whose position_type is Goalkeeper. The team attacking
    left_to_right in the first half defends the left goal; the teams swap ends at halftime.
    """
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


def read_features(raw_path, meta_path, n_frames, pitch):
    """Per frame: ball position (m), and the distance from the ball to each team's nearest player.

    home_fixed, away_fixed     teams from the players' slots, as the baseline has them
    home_keeper, away_keeper   the same, but a goalkeeper takes the team defending his side
    player_speed               median speed of the tracked players, m/s, from frame to frame
    NaN where the ball, or that team, is not tracked.
    """
    length, width = pitch
    halfway = length / 2.0
    teams = team_of_each_track(meta_path)
    keepers, left_first_half = keeper_sides(meta_path)
    blocks = half_video_blocks(GAME)
    second_half_from = (blocks[0][2] + blocks[1][1]) // 2  # middle of halftime

    out = {name: np.full(n_frames, np.nan) for name in
           ("ball_x", "ball_y", "home_fixed", "away_fixed", "home_keeper", "away_keeper",
            "player_speed")}
    fps = GAMES[GAME]["fps"]
    previous_frame, previous = None, {}

    with open(raw_path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1  # frame_id counts from 1, submissions from 0
            if not 0 <= frame < n_frames:
                continue
            # The players' speed is read on every frame, ball or no ball.
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
            previous_frame, previous = frame, positions
            if parts[2][0] == MISSING:
                continue
            ball = parts[2].split(",")
            bx, by = float(ball[0]) * length, float(ball[1]) * width
            out["ball_x"][frame], out["ball_y"][frame] = bx, by

            left_team = left_first_half if frame < second_half_from else (
                AWAY if left_first_half == HOME else HOME)
            fixed = [None, None]
            keeper = [None, None]
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if group[0] == MISSING:
                    continue
                values = group.split(",")
                px, py = float(values[0]) * length, float(values[1]) * width
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


def load_features(data_dir):
    """read_features(), cached next to the data so the slow pass runs once."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, data_dir)
    cache = os.path.join(os.path.dirname(raw_path), CACHE_NAME.format(GAME))
    if os.path.isfile(cache) and os.path.getmtime(cache) >= os.path.getmtime(raw_path):
        with np.load(cache) as stored:
            return {key: stored[key] for key in stored.files}
    print("reading {} (once; cached afterwards) ...".format(os.path.basename(raw_path)))
    features = read_features(raw_path, meta_path, source_frames("tactical", GAME),
                             GAMES[GAME]["pitch"])
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


# -- the rules ------------------------------------------------------------------------------------

def ball_kinematics(x, y, fps):
    """(speed, smoothed speed) in m/s, NaN where the ball is not seen on both ends."""
    k = 2
    speed = np.full(len(x), np.nan)
    speed[k:-k] = np.hypot(x[2 * k:] - x[:-2 * k], y[2 * k:] - y[:-2 * k]) * fps / (2.0 * k)
    smooth = pd.Series(speed).rolling(SMOOTH_FRAMES, center=True,
                                      min_periods=SMOOTH_FRAMES // 2).median().to_numpy()
    return speed, smooth


def path_changes(x, y, fps):
    """True where the ball's path turns or changes speed (rule 2), from PATH_SPAN frames each side."""
    k = PATH_SPAN
    n = len(x)
    change = np.zeros(n, dtype=bool)
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
    change = np.nan_to_num(turn, nan=0).astype(bool) | np.nan_to_num(speed_change, nan=0).astype(bool)
    return change


def runs_of(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def first_true(mask, start, length):
    hits = np.flatnonzero(mask[start:start + length])
    return start + int(hits[0]) if len(hits) else None


def smart_tagging(data_dir):
    """(kickoff tag frames per half, [(corner tag frame, team)], [after-goal kickoff tag frames],
    [(free-kick tag frame, team)], [(goal-kick tag frame, team)], [(throw-in tag frame, team)])."""
    meta_path = require(["atd_meta"], GAME, data_dir)[0]
    home, away = (name.strip().lower() for name in team_names(meta_path))
    teams = {home: HOME, away: AWAY}
    kickoffs, corners, goals, free_kicks, goal_kicks, throw_ins = {}, [], [], [], [], []
    for event in load_smart_tagging.load(GAME, data_dir):
        kind = (event["tags"].get("Type") or "").strip().lower()
        if kind == "kick off (start)":
            half = 1 if event["tags"].get("Half") == "1st Half" else 2
            kickoffs[half] = event["start_frame"]
        elif kind == "kick off (after goal)":
            goals.append(event["start_frame"])
        elif kind in ("corner kick", "free kick", "goal kick", "goal-kick", "throw in", "throw-in"):
            team = teams.get((event["tags"].get("Team") or "").strip().lower(), NOBODY)
            if kind == "corner kick":
                corners.append((event["start_frame"], team))
            elif kind == "free kick":
                free_kicks.append((event["start_frame"], team))
            elif kind in ("goal kick", "goal-kick"):
                goal_kicks.append((event["start_frame"], team))
            else:
                throw_ins.append((event["start_frame"], team))
    return (kickoffs, sorted(corners), sorted(goals), sorted(free_kicks),
            sorted(goal_kicks), sorted(throw_ins))


def label_half(first, last, f, kick_tag, corners, left_team, report, ownership="nearest",
               touch_radius=TOUCH_RADIUS, placed_seconds=PLACED_SECONDS, launch_rule=True,
               goals=(), flight=True, backdate=True, free_kicks=(), slow_stop=False,
               merge_gap=0.0, merge_tempo=MERGE_TEMPO, merge_max_ball=None,
               goal_kicks=(), throw_ins=()):
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

    # Rule 2: on which frames does the nearest player's team get the ball?
    with np.errstate(invalid="ignore"):
        within = seen & (nearest <= touch_radius)
    if ownership == "touch":
        with np.errstate(invalid="ignore"):
            touch = f["changes"] & within
            close = within & (speed < CONTROL_SPEED)
        control = np.zeros(len(x), dtype=bool)
        starts, ends = runs_of(close)
        for start, end in zip(starts, ends):
            if end - start >= CONTROL_FRAMES:
                control[start:end] = True
        gets_it = touch | control
    else:
        gets_it = within
    if flight:
        with np.errstate(invalid="ignore"):
            gets_it = gets_it & ~(speed > FLIGHT_SPEED)

    # Rules 4 and 5: exits over a goal line or a touchline, and their restarts.
    with np.errstate(invalid="ignore"):
        over = seen & ((x < 0) | (x > length))
        over_side = seen & ((y < 0) | (y > width)) & ~over
        in_box = seen & (np.minimum(x, length - x) <= BOX_DEPTH) & (
            np.abs(y - width / 2.0) <= BOX_HALF_WIDTH)
        placed_ball = in_box & (smooth < PLACED_SPEED)
    placed_frames = max(1, int(round(placed_seconds * fps)))
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
        if restart is None:
            placed_runs = runs_of(placed_ball[end:end + GOAL_KICK_SEARCH])
            for p_start, p_end in zip(*placed_runs):
                if p_end - p_start >= placed_frames:
                    hit = first_true(seen & (speed > KICK_SPEED), end + p_end - SMOOTH_FRAMES // 2,
                                     GOAL_KICK_SEARCH)
                    if hit is not None:
                        restart = (hit, defending, "goal kick")
                    break
        if restart is None:
            hit = first_true(gets_it, end, len(x))
            restart = (hit if hit is not None else last + 1, None, FALLBACK)
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
    stoppages = {}  # start frame -> (kick frame, team); may begin inside another dead stretch
    for tag, tag_team in free_kicks:
        if not max(first, kick) <= tag <= last:
            continue
        kick_f = first_true(seen & (speed > KICK_SPEED), tag, CORNER_KICK_SEARCH)
        if kick_f is None:
            continue
        start = tag
        while start > max(first, kick, tag - FK_LOOKBACK) and quiet[start - 1]:
            if seen[start - 1] and (smooth[start - 1] >= ACTIVE_BALL_SPEED
                                    or speed[start - 1] >= KICK_SPEED):
                break
            start -= 1
        stoppages[start] = (kick_f, tag_team if tag_team != NOBODY else None)
        report["free kick"] = report.get("free kick", 0) + 1

    # Tagged goal kicks and throw-ins: dead from the preceding quiet-ball stretch to the kick.
    for tags, name in ((goal_kicks, "goal kick tag"), (throw_ins, "throw-in tag")):
        for tag, tag_team in tags:
            if not max(first, kick) <= tag <= last:
                continue
            kick_t = first_true(seen & (speed > KICK_SPEED), tag, CORNER_KICK_SEARCH)
            if kick_t is None:
                continue
            start = tag
            floor = max(first, kick, tag - FK_LOOKBACK)
            while start > floor and quiet[start - 1]:
                if seen[start - 1] and (smooth[start - 1] >= ACTIVE_BALL_SPEED
                                        or speed[start - 1] >= KICK_SPEED):
                    break
                start -= 1
            stoppages[start] = (kick_t, tag_team if tag_team != NOBODY else None)
            report[name] = report.get(name, 0) + 1

    # Rule 8: untagged stoppages -- the players walking while the ball is quiet.
    slow_stops = {}  # start frame -> first frame after the stretch
    restart_kicks = np.zeros(len(x), dtype=bool)  # rule 8a
    if slow_stop:
        with np.errstate(invalid="ignore"):
            walking = (f["tempo"] < SLOW_TEMPO) & quiet
            if RESTART_CUT:
                still = seen & (smooth < PLACED_SPEED)
                rearm = np.zeros(len(x), dtype=bool)
                for s_, e_ in zip(*runs_of(still[first:last + 1])):
                    if e_ - s_ >= int(round(RESTART_STILL * fps)):
                        hit = first_true(seen & (speed > RESTART_KICK),
                                         first + e_ - SMOOTH_FRAMES // 2, RESTART_SEARCH)
                        if hit is not None:
                            restart_kicks[hit] = True
                            j = min(len(x), hit + int(round(REARM_SECONDS * fps)))
                            stop = min(len(x), hit + int(round(REARM_MAX * fps)))
                            while j < stop and not f["tempo"][j] >= SLOW_TEMPO:
                                j += 1
                            rearm[hit:j] = True
                walking = walking & ~rearm  # a stoppage ends at its restart kick
                report["restart kicks"] = int(restart_kicks[first:last + 1].sum())
        starts, ends = runs_of(walking[first:last + 1])
        for start, end in zip(starts + first, ends + first):
            if start >= kick and end - start >= int(round(SLOW_SECONDS * fps)):
                if SLOW_BACK is not None:  # rule 8b: back to where the tempo last stood at SLOW_BACK
                    floor_ = max(first, kick, start - SLOW_BACK_MAX)
                    kicks_before = np.flatnonzero(restart_kicks[floor_:start])
                    if len(kicks_before):  # never back across a restart
                        floor_ = floor_ + int(kicks_before[-1]) + 1
                    while start > floor_ and not f["tempo"][start - 1] >= SLOW_BACK:
                        if seen[start - 1] and (smooth[start - 1] >= ACTIVE_BALL_SPEED
                                                or speed[start - 1] >= KICK_SPEED):
                            break
                        start -= 1
                slow_stops[start] = max(end, slow_stops.get(start, end))
        report["slow stop"] = len(slow_stops)

    # Rule 3: launches -- the ball's speed rising past LAUNCH_SPEED inside a penalty area.
    with np.errstate(invalid="ignore"):
        fast = speed >= LAUNCH_SPEED
    launch = np.zeros(len(x), dtype=bool)
    launch[1:] = fast[1:] & ~fast[:-1] & in_box[1:]
    right_team = AWAY if left_team == HOME else HOME

    # The pass over the half.
    team = near_team[kick] if kick <= last and np.isfinite(nearest[kick]) else NOBODY
    dead_until, restart_team, flight_from = kick, team, None
    launches, floor, last_touch = 0, kick, {HOME: -1, AWAY: -1}
    for t in range(first, last + 1):
        if t in restarts and t >= dead_until:
            dead_until, restart_team, _ = restarts[t]
            team, flight_from = NOBODY, None
        if t in stoppages and stoppages[t][0] > dead_until:
            dead_until, restart_team = stoppages[t]
            team, flight_from = NOBODY, None
            labels[t - first] = "D"
        if t in slow_stops and (t >= dead_until or (SLOW_BACK is not None
                                                    and slow_stops[t] > dead_until)):
            dead_until, restart_team = slow_stops[t], None  # restarts with the next touch
            team, flight_from = NOBODY, None
        if t < dead_until:
            continue
        if t == dead_until:
            floor = t  # no change of team is moved back past a restart
            if restart_team not in (None, NOBODY):
                team = restart_team
        if launch_rule and launch[t] and team != NOBODY and flight_from is None:
            if team == (left_team if x[t] < length / 2.0 else right_team):  # its own box
                flight_from = t
                launches += 1
        if flight_from is not None:
            # The keeper is still at the ball as it leaves: the first touch is the first frame,
            # after the ball has left him (nobody within TOUCH_RADIUS), that someone touches it.
            left_keeper = np.any(~within[flight_from:t + 1])
            if not (within[t] and left_keeper):
                labels[t - first] = LABEL[team]
                continue
            flight_from, floor = None, t  # the launch stays with the keeper's team until here
        if gets_it[t]:
            toucher = near_team[t]
            if (team not in (NOBODY, toucher)
                    and np.isfinite(speed[t])
                    and speed[t] > FLIGHT_SPEED
                    and not f["changes"][t]):
                labels[t - first] = LABEL[team]
                continue
            if backdate and team not in (NOBODY, toucher):
                start = max(last_touch[team] + 1, t - BACKDATE_MAX, floor)
                span = labels[start - first:t - first]
                span[span == LABEL[team]] = LABEL[toucher]
            team = toucher
            last_touch[team] = t
        if team != NOBODY:
            labels[t - first] = LABEL[team]
    report["launches"] = launches

    # Rule 9: join dead stretches at most merge_gap seconds apart.
    if merge_gap > 0:
        gap_frames, merged = int(round(merge_gap * fps)), 0
        starts, ends = runs_of(labels != "D")
        for start, end in zip(starts, ends):
            if start <= kick - first or end >= n or end - start > gap_frames:
                continue  # not between two dead stretches, or too long
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                walk = np.nanmean(f["tempo"][first + start:first + end])
                fastest = np.nanmax(smooth[first + start:first + end])
            if restart_kicks[first + start:first + end].any():
                continue  # rule 8a: play restarted in the gap
            if merge_max_ball is not None and fastest > merge_max_ball:
                continue  # the ball was kicked in the gap: play restarted
            if np.isnan(walk) or walk < merge_tempo:
                labels[start:end] = "D"
                merged += 1
        report["merged gaps"] = merged
    return labels


def possessions(labels):
    starts = np.concatenate(([0], np.flatnonzero(labels[1:] != labels[:-1]) + 1))
    return int(np.isin(labels[starts], ("H", "A")).sum())


def suppress_flickers(labels, max_frames=8):
    """Absorb brief opponent takeovers flanked by the other team on both sides."""
    for team_char, opp_char in (("H", "A"), ("A", "H")):
        runs_edges = np.diff(np.concatenate(([0], (labels == team_char).astype(np.int8), [0])))
        starts = np.flatnonzero(runs_edges == 1)
        ends = np.flatnonzero(runs_edges == -1)
        for start, end in zip(starts, ends):
            if end - start >= max_frames:
                continue
            if start > 0 and end < len(labels) and labels[start - 1] == opp_char and labels[end] == opp_char:
                labels[start:end] = opp_char
    return labels


def prepare(data_dir, keepers=True):
    """Everything the rules read, once: (tracking arrays, kickoff tags, corner tags, goal tags,
    team defending the left goal in the first half). Shared by main() and sweep_radius.py."""
    fps = GAMES[GAME]["fps"]
    features = load_features(data_dir)
    # Rule 0: "_keeper" distances put each goalkeeper slot in the team defending its side.
    suffix = "_keeper" if keepers else "_fixed"
    f = {"ball_x": features["ball_x"], "ball_y": features["ball_y"],
         "home": np.nan_to_num(features["home" + suffix], nan=np.inf),
         "away": np.nan_to_num(features["away" + suffix], nan=np.inf)}
    f["speed"], f["smooth"] = ball_kinematics(f["ball_x"], f["ball_y"], fps)
    f["changes"] = path_changes(f["ball_x"], f["ball_y"], fps)
    f["tempo"] = tempo(features)  # rule 8
    kickoffs, corners, goals, free_kicks, goal_kicks, throw_ins = smart_tagging(data_dir)
    meta_path = require(["atd_meta"], GAME, data_dir)[0]
    _, left_first_half = keeper_sides(meta_path)  # only which side each team defends
    return f, kickoffs, corners, goals, left_first_half, free_kicks, goal_kicks, throw_ins


def label_match(prepared, ownership="nearest", touch_radius=TOUCH_RADIUS,
                placed_seconds=PLACED_SECONDS, launch_rule=True, flight=True, backdate=True,
                free_kick_rule=True, slow_stop=False, merge_gap=0.0, merge_tempo=MERGE_TEMPO,
                merge_max_ball=None, reports=None):
    """The full-length label array for one set of settings. reports, if a list, gets one per half."""
    f, kickoffs, corners, goals, left_first_half, free_kicks, goal_kicks, throw_ins = prepared
    labels = np.full(len(f["ball_x"]), "X", dtype="<U1")
    for half, first, last in half_video_blocks(GAME):
        left_team = left_first_half if half == 1 else (AWAY if left_first_half == HOME else HOME)
        report = {}
        labels[first:last + 1] = label_half(first, last, f, kickoffs.get(half), corners,
                                            left_team, report, ownership, touch_radius,
                                            placed_seconds, launch_rule, goals, flight, backdate,
                                            free_kicks if free_kick_rule else (), slow_stop,
                                            merge_gap, merge_tempo, merge_max_ball,
                                            goal_kicks, throw_ins)
        labels[first:last + 1] = suppress_flickers(labels[first:last + 1])
        if reports is not None:
            reports.append((half, half in kickoffs, report))
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ownership", choices=("nearest", "touch"), default="nearest",
                        help="rule 2: the nearest player, or a change of the ball's path at a player")
    parser.add_argument("--nearest-radius", type=float, default=TOUCH_RADIUS, metavar="M",
                        help="the touch distance, everywhere (default 0.4 m)")
    parser.add_argument("--no-keepers", action="store_true",
                        help="turn off rule 0: goalkeepers keep the team of their ATD slot")
    parser.add_argument("--placed-seconds", type=float, default=PLACED_SECONDS, metavar="S",
                        help="rule 5: how long the ball must stand still on the pitch before a goal kick")
    parser.add_argument("--no-launch", action="store_true",
                        help="turn off rule 3, the goalkeeper's throw or kick")
    parser.add_argument("--no-flight", action="store_true",
                        help="rule 2: let a ball faster than FLIGHT_SPEED be touched")
    parser.add_argument("--no-backdate", action="store_true",
                        help="rule 2: a change of team starts at the touch, not when the ball left")
    parser.add_argument("--no-slow-stop", action="store_true",
                        help="turn off rule 8 (and so 8a, 8b): players walking + ball quiet")
    parser.add_argument("--slow-tempo", type=float, default=SLOW_TEMPO, metavar="M/S",
                        help="rule 8: the tempo line (default 1.2)")
    parser.add_argument("--slow-seconds", type=float, default=SLOW_SECONDS, metavar="S",
                        help="rule 8: how long (default 1.0)")
    parser.add_argument("--merge-gap", type=float, default=5.0, metavar="S",
                        help="rule 9: join dead stretches at most S seconds apart (default 5; 0 = off)")
    parser.add_argument("--merge-tempo", type=float, default=MERGE_TEMPO, metavar="M/S",
                        help="rule 9: only if the players walk the gap (default 1.5; 99 = always)")
    parser.add_argument("--merge-max-ball", type=float, default=None, metavar="M/S",
                        help="rule 9: not across a gap in which the ball goes faster than this")
    parser.add_argument("--restart-cut", action="store_true",
                        help="rule 8a: an untagged stoppage ends at a placed-ball kick")
    parser.add_argument("--restart-still", type=float, default=RESTART_STILL, metavar="S",
                        help="rule 8a: how long the ball stands still before the kick (default 1.0)")
    parser.add_argument("--restart-kick", type=float, default=RESTART_KICK, metavar="M/S",
                        help="rule 8a: how fast the kick (default 5)")
    parser.add_argument("--slow-back", type=float, default=1.8, metavar="M/S",
                        help="rule 8b: start an untagged stoppage where the tempo was last this high "
                             "(default 1.8; 0 = off)")
    parser.add_argument("--no-free-kicks", action="store_true",
                        help="turn off rule 7, the Smart Tagging free kicks")
    parser.add_argument("--data-dir", default=None, help="data bundle, if not ./data or MTAH_DATA_DIR")
    parser.add_argument("--out", default=RESULTS_DIR, help="where to write")
    args = parser.parse_args()
    if args.slow_back is not None and args.slow_back <= 0:
        args.slow_back = None
    globals().update(SLOW_TEMPO=args.slow_tempo, SLOW_SECONDS=args.slow_seconds,  # rule 8 reads these
                     RESTART_CUT=args.restart_cut, RESTART_STILL=args.restart_still,
                     RESTART_KICK=args.restart_kick, SLOW_BACK=args.slow_back)

    reports = []
    labels = label_match(prepare(args.data_dir, not args.no_keepers), args.ownership,
                         args.nearest_radius, args.placed_seconds, not args.no_launch,
                         not args.no_flight, not args.no_backdate, not args.no_free_kicks,
                         not args.no_slow_stop, args.merge_gap, args.merge_tempo, args.merge_max_ball,
                         reports=reports)
    blocks = {half: (first, last) for half, first, last in half_video_blocks(GAME)}
    for half, tagged, report in reports:
        part = labels[blocks[half][0]:blocks[half][1] + 1]
        kickoff = report.pop("kickoff")
        launches = report.pop("launches", 0)
        print("half {}: kickoff at {}{}   restarts: {}   keeper launches {}   "
              "possessions {}   dead {:.0f}%".format(
                  half, kickoff, "" if tagged else " (no tag: half start)",
                  ", ".join("{} {}".format(v, k) for k, v in report.items()) or "none",
                  launches, possessions(part), 100.0 * np.mean(part == "D")))

    name = "best_so_far" if len(sys.argv) == 1 else "best_so_far_custom"
    path = write_submission(os.path.join(args.out, "{}.csv".format(name)), labels)
    print("\n{}\nscore it:  python evaluation/run_evaluation.py --bootstrap 1000".format(
        os.path.relpath(path, ROOT)))


if __name__ == '__main__':
    main()
