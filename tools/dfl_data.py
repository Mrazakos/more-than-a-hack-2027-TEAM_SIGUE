"""The DFL open dataset (Bundesliga 2022/23, CC BY 4.0, DFL / Sportec) as a labeller input.

External tuning data only: nothing here reads BAR-ATH labels or video. Per match it gives

  * `ground_truth`   H / A / D per frame from the tracking's own BallPossession and BallStatus
                     (BallStatus 1 = alive, otherwise dead; BallPossession 1 = home, 2 = away);
  * `build_match`    a `starter/possession/context.MatchData` with the same per-frame arrays the ATD
                     reader makes (ball in metres, nearest player per team, tempo, width), plus
                     stand-ins for the Smart Tagging events built from the DFL event file.

Timeline: the two halves are laid end to end with GAP unlabelled (X) frames between them, so no
possession is scored across half time. Coordinates: DFL is centred on the pitch, metres; the
labeller wants 0..length, 0..width, so x += length / 2 and y += width / 2.

    python tools/dfl_data.py            # parse every match once (cached) and print sanity numbers
"""

import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "starter"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

from baseline import AWAY, HOME, NOBODY  # noqa: E402
from possession.context import MatchData  # noqa: E402
from possession.signals import ball_kinematics, path_changes, tempo  # noqa: E402
from possession.tags import SmartTags  # noqa: E402

DFL_DIR = os.environ.get("MTAH_DFL_DIR", os.path.join(ROOT, "data_external", "dfl"))
CACHE_DIR = os.path.join(DFL_DIR, "cache")
FPS = 25
GAP = 500                  # X frames between the halves
MAX_PLAYER_SPEED = 12.0    # m/s, as tracking.py: a faster jump is a glitch
TAG_LEAD = 85              # frames: a Smart Tagging kickoff tag opens ~3.5 s before the kick (params)
CACHE_VERSION = 1

MATCHES = [
    ("DFL-COM-000001", "DFL-MAT-J03WMX"), ("DFL-COM-000001", "DFL-MAT-J03WN1"),
    ("DFL-COM-000002", "DFL-MAT-J03WOH"), ("DFL-COM-000002", "DFL-MAT-J03WOY"),
    ("DFL-COM-000002", "DFL-MAT-J03WPY"), ("DFL-COM-000002", "DFL-MAT-J03WQQ"),
    ("DFL-COM-000002", "DFL-MAT-J03WR9"),
]
MATCH_IDS = [m for _, m in MATCHES]


def _path(kind, comp, match):
    return os.path.join(DFL_DIR, "DFL_{}_{}_{}.xml".format(kind, comp, match))


def paths(match):
    comp = dict((m, c) for c, m in MATCHES)[match]
    return (_path("02_01_matchinformation", comp, match), _path("03_02_events_raw", comp, match),
            _path("04_03_positions_raw_observed", comp, match))


def _epoch(text):
    return datetime.fromisoformat(text).timestamp()


# -- parsing -------------------------------------------------------------------------------------

def read_match_information(path):
    root = ET.parse(path).getroot()
    general = root.find(".//General")
    env = root.find(".//Environment")
    teams = {}
    for team in root.iter("Team"):
        teams[team.get("Role")] = team.get("TeamId")
    keepers = {p.get("PersonId") for p in root.iter("Player") if p.get("PlayingPosition") == "TW"}
    return {"home": general.get("HomeTeamId"), "away": general.get("GuestTeamId"),
            "pitch": (float(env.get("PitchX")), float(env.get("PitchY"))), "keepers": keepers}


def read_positions(path):
    """{half: {"ball": dict of arrays, "players": {(team, person): (x, y) arrays}}} in raw DFL units.

    A FrameSet is one person for one half; every Frame line is `N T X Y ...`, the ball's also ends
    in BallPossession and BallStatus. Parsed by splitting on quotes: ~100x faster than a DOM."""
    halves = {}
    section = team = person = None
    store = None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("<FrameSet"):
                parts = line.split('"')
                section, team, person = parts[1], parts[5], parts[7]
                store = []
            elif line.startswith("<Frame "):
                parts = line.split('"')
                try:
                    row = [int(parts[1]), parts[3], float(parts[5]), float(parts[7])]
                except ValueError:
                    continue
                if team == "BALL":
                    row += [int(parts[-4]), int(parts[-2])]
                store.append(row)
            elif line.startswith("</FrameSet"):
                if team != "referee" and store:
                    halves.setdefault(section, {})[(team, person)] = store
                store = None
    return halves


def _half_arrays(frame_sets, n0, n1, length, width):
    """Arrays over frames n0..n1 for the ball and for each person (NaN where not tracked)."""
    n = n1 - n0 + 1
    ball_rows = next(rows for (team, _), rows in frame_sets.items() if team == "BALL")
    ball = {k: np.full(n, np.nan) for k in ("x", "y", "t")}
    ball["status"] = np.zeros(n, dtype=np.int8)
    ball["poss"] = np.zeros(n, dtype=np.int8)
    for number, stamp, x, y, poss, status in ball_rows:
        i = number - n0
        if 0 <= i < n:
            ball["x"][i], ball["y"][i], ball["t"][i] = x + length / 2.0, y + width / 2.0, _epoch(stamp)
            ball["poss"][i], ball["status"][i] = poss, status
    players = {}
    for (team, person), rows in frame_sets.items():
        if team == "BALL":
            continue
        px, py = np.full(n, np.nan), np.full(n, np.nan)
        for number, _stamp, x, y in rows:
            i = number - n0
            if 0 <= i < n:
                px[i], py[i] = x + length / 2.0, y + width / 2.0
        players[(team, person)] = (px, py)
    return ball, players


def parse_match(match, refresh=False):
    """Everything one match needs, cached as .npz: per global frame arrays plus the metadata."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, "{}_v{}.npz".format(match, CACHE_VERSION))
    info_path, _events_path, pos_path = paths(match)
    if os.path.isfile(cache) and not refresh and os.path.getmtime(cache) >= os.path.getmtime(pos_path):
        with np.load(cache, allow_pickle=True) as stored:
            return {k: stored[k] for k in stored.files}
    info = read_match_information(info_path)
    length, width = info["pitch"]
    raw = read_positions(pos_path)
    first_n = {}
    sections = [s for s in ("firstHalf", "secondHalf") if s in raw]
    for s in sections:
        ball_rows = next(rows for (team, _), rows in raw[s].items() if team == "BALL")
        first_n[s] = (ball_rows[0][0], ball_rows[-1][0])
    parts, blocks, offset = [], [], 0
    persons = sorted({(team, person) for s in sections for (team, person) in raw[s] if team != "BALL"})
    out = {k: [] for k in ("ball_x", "ball_y", "ball_t", "status", "poss")}
    pos_x = {key: [] for key in persons}
    pos_y = {key: [] for key in persons}
    for half_no, s in enumerate(sections, start=1):
        n0, n1 = first_n[s]
        ball, players = _half_arrays(raw[s], n0, n1, length, width)
        n = n1 - n0 + 1
        for key, name in (("x", "ball_x"), ("y", "ball_y"), ("t", "ball_t"), ("status", "status"),
                          ("poss", "poss")):
            out[name].append(ball[key])
        for key in persons:
            px, py = players.get(key, (np.full(n, np.nan), np.full(n, np.nan)))
            pos_x[key].append(px)
            pos_y[key].append(py)
        blocks.append((half_no, offset, offset + n - 1))
        offset += n
        if half_no < len(sections):  # the half-time gap, unlabelled
            gap_nan = np.full(GAP, np.nan)
            out["ball_x"].append(gap_nan); out["ball_y"].append(gap_nan); out["ball_t"].append(gap_nan)
            out["status"].append(np.zeros(GAP, dtype=np.int8)); out["poss"].append(np.zeros(GAP, dtype=np.int8))
            for key in persons:
                pos_x[key].append(gap_nan); pos_y[key].append(gap_nan)
            offset += GAP
    result = {k: np.concatenate(v) for k, v in out.items()}
    result["players_x"] = np.stack([np.concatenate(pos_x[k]) for k in persons], axis=1)
    result["players_y"] = np.stack([np.concatenate(pos_y[k]) for k in persons], axis=1)
    result["player_team"] = np.array([HOME if team == info["home"] else AWAY for team, _ in persons])
    result["player_keeper"] = np.array([person in info["keepers"] for _, person in persons])
    result["blocks"] = np.array(blocks)
    result["pitch"] = np.array([length, width])
    np.savez_compressed(cache, **result)
    return result


# -- ground truth --------------------------------------------------------------------------------

def ground_truth(m):
    """H / A / D per frame from the tracking's BallStatus / BallPossession; X in the gap."""
    labels = np.full(len(m["ball_x"]), "X", dtype="<U1")
    for _half, first, last in m["blocks"]:
        status, poss = m["status"][first:last + 1], m["poss"][first:last + 1]
        span = np.full(last - first + 1, "D", dtype="<U1")
        span[(status == 1) & (poss == 1)] = "H"
        span[(status == 1) & (poss == 2)] = "A"
        labels[first:last + 1] = span
    return labels


def controlled(labels, min_seconds):
    """The ground truth with 'a simple touch of the ball without control' removed (CHALLENGE.md: a
    possession needs the ball for 'a few seconds'): an H or A spell shorter than min_seconds, with
    the other team's possession directly on both sides, goes to that team. Repeated until stable.
    The 2 s is fixed from that wording, not from the BAR-ATH labels."""
    labels = labels.copy()
    limit = int(round(min_seconds * FPS))
    changed = limit > 0
    while changed:
        changed = False
        starts = np.concatenate(([0], np.flatnonzero(labels[1:] != labels[:-1]) + 1))
        ends = np.concatenate((starts[1:], [len(labels)]))
        for i in range(1, len(starts) - 1):
            here = labels[starts[i]]
            before, after = labels[starts[i - 1]], labels[starts[i + 1]]
            if (here in "HA" and before in "HA" and before == after and before != here
                    and ends[i] - starts[i] < limit):
                labels[starts[i]:ends[i]] = before
                changed = True
    return labels


# -- the labeller's inputs -----------------------------------------------------------------------

def _nearest_by_team(m, bx, by):
    out = {}
    for team in (HOME, AWAY):
        cols = m["player_team"] == team
        d = np.hypot(m["players_x"][:, cols] - bx[:, None], m["players_y"][:, cols] - by[:, None])
        with np.errstate(all="ignore"):
            nearest = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1)
        out[team] = nearest  # inf where nobody is tracked or the ball is not seen
    return out


def _player_speed(m):
    """Median player speed per frame from position differences (as tracking.py does)."""
    dx = np.diff(m["players_x"], axis=0, prepend=np.nan)
    dy = np.diff(m["players_y"], axis=0, prepend=np.nan)
    step = np.hypot(dx, dy) * FPS
    step[step > MAX_PLAYER_SPEED] = np.nan
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(step, axis=1)


def _width_diff(m, params):
    widths = {}
    for team in (HOME, AWAY):
        cols = (m["player_team"] == team) & ~m["player_keeper"]
        ys = m["players_y"][:, cols]
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                w = np.nanstd(ys, axis=1)
        w[np.sum(~np.isnan(ys), axis=1) < 5] = np.nan
        widths[team] = w
    return pd.Series(widths[HOME] - widths[AWAY]).rolling(
        params.shape_smooth, center=True, min_periods=params.shape_smooth // 2).mean().to_numpy()


def _frame_of(m, stamp):
    """Global frame of a wall-clock timestamp: the nearest ball frame in time."""
    t = m["ball_t"]
    valid = np.flatnonzero(~np.isnan(t))
    i = np.searchsorted(t[valid], stamp)
    i = min(max(i, 0), len(valid) - 1)
    return int(valid[i])


def read_tags(m, match):
    """Smart Tagging stand-ins from the DFL event file (see SmartTags in possession/tags.py):
    kickoffs {half: tag frame} (opening TAG_LEAD frames before the kick, as the real tag does),
    goals (kickoff after a goal), corners and free kicks as (frame the decision was taken, team)."""
    info = read_match_information(paths(match)[0])
    teams = {info["home"]: HOME, info["away"]: AWAY}
    root = ET.parse(paths(match)[1]).getroot()
    kickoffs, goals, corners, free_kicks = {}, [], [], []
    for event in root.iter("Event"):
        for child in event:
            kind = child.tag
            if kind == "KickOff":
                frame = _frame_of(m, _epoch(event.get("EventTime")))
                section = child.get("GameSection")
                if section:
                    half = 1 if section == "firstHalf" else 2
                    kickoffs[half] = max(frame - TAG_LEAD, int(m["blocks"][half - 1][1]))
                else:
                    goals.append(max(frame - TAG_LEAD, 0))
            elif kind in ("CornerKick", "FreeKick"):
                stamp = child.get("DecisionTimestamp") or event.get("EventTime")
                item = (_frame_of(m, _epoch(stamp)), teams.get(child.get("Team"), NOBODY))
                (corners if kind == "CornerKick" else free_kicks).append(item)
    return SmartTags(kickoffs, sorted(corners), sorted(goals), sorted(free_kicks))


def build_match(m, match, params, ball_mask=None):
    """A MatchData for the labeller. ball_mask (bool per frame, True = ball seen) hides the ball
    the way the ATD does (degrade.context_mask)."""
    bx, by = m["ball_x"].copy(), m["ball_y"].copy()
    if ball_mask is not None:
        bx[~ball_mask] = np.nan
        by[~ball_mask] = np.nan
    near = _nearest_by_team(m, bx, by)
    speed, smooth = ball_kinematics(bx, by, FPS, params)
    # Half 1 attacks which way? The kickoff event names the team on the left (x small).
    root = ET.parse(paths(match)[1]).getroot()
    left_first_half = HOME
    info = read_match_information(paths(match)[0])
    for kick in root.iter("KickOff"):
        if kick.get("GameSection") == "firstHalf":
            left_first_half = HOME if kick.get("TeamLeft") == info["home"] else AWAY
            break
    return MatchData(
        ball_x=bx, ball_y=by, home=near[HOME], away=near[AWAY], speed=speed, smooth=smooth,
        changes=path_changes(bx, by, FPS, params),
        tempo=tempo(_player_speed(m), params), width_diff=_width_diff(m, params),
        tags=read_tags(m, match), left_first_half=left_first_half,
        pitch=(float(m["pitch"][0]), float(m["pitch"][1])), fps=FPS,
        blocks=[tuple(int(v) for v in b) for b in m["blocks"]])


# -- sanity numbers ------------------------------------------------------------------------------

def summary(m):
    gt = ground_truth(m)
    graded = gt != "X"
    starts = np.concatenate(([0], np.flatnonzero(gt[1:] != gt[:-1]) + 1))
    spells = [(gt[s], e - s) for s, e in zip(starts, list(starts[1:]) + [len(gt)]) if gt[s] in "HA"]
    return {"frames": int(graded.sum()), "dead_share": float(np.mean(gt[graded] == "D")),
            "home_share_of_live": float(np.mean(gt[gt != "D"][gt[gt != "D"] != "X"] == "H")),
            "possessions": len(spells),
            "mean_possession_s": float(np.mean([n for _, n in spells]) / FPS),
            "ball_seen": float(np.mean(~np.isnan(m["ball_x"][graded])))}


if __name__ == "__main__":
    from possession.config import Params
    for match in MATCH_IDS:
        if not os.path.isfile(paths(match)[2]):
            print(match, "positions file missing")
            continue
        m = parse_match(match)
        print(match, {k: round(v, 3) if isinstance(v, float) else v for k, v in summary(m).items()})
        tags = read_tags(m, match)
        print("   tags: kickoffs", tags.kickoffs, "corners", len(tags.corners), "goals", len(tags.goals),
              "free kicks", len(tags.free_kicks))
