"""Smart Tagging events the rules use, as one typed object."""

from typing import List, NamedTuple, Tuple

from games import GAME, require
from baseline import HOME, AWAY, NOBODY, team_names
import load_smart_tagging


class SmartTags(NamedTuple):
    kickoffs: dict                      # half (1 or 2) -> video frame of the "Kick Off (Start)" tag
    corners: List[Tuple[int, int]]      # (tag frame, team), sorted
    goals: List[int]                    # "Kick Off (After Goal)" tag frames, sorted
    free_kicks: List[Tuple[int, int]]   # (tag frame, team), sorted


def load_tags(data_dir):
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
    return SmartTags(kickoffs, sorted(corners), sorted(goals), sorted(free_kicks))
