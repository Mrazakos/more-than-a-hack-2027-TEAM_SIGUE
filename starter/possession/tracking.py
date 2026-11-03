"""Reading the ATD into per-frame arrays, with a cache next to the data (the slow pass runs once)."""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from games import GAME, GAMES, half_video_blocks, require, source_frames
from baseline import AWAY, FIELD_SEP, GROUP_SEP, HOME, MISSING, team_of_each_track

CACHE_NAME = "best_so_far_features_v4_{}.npz"
MAX_PLAYER_SPEED = 12.0  # m/s; a faster jump between frames is an identity switch (part of the cache)


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
