"""What the rules read: MatchData (everything, once) and HalfContext (one half's view of it)."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from games import GAME, GAMES, require
from baseline import AWAY, HOME

from .signals import ball_kinematics, first_true, path_changes, tempo
from .tags import SmartTags, load_tags
from .tracking import keeper_sides, load_features


def opponent(team):
    return AWAY if team == HOME else HOME


@dataclass
class MatchData:
    """Per-frame arrays for the whole match (video frames) plus the tags. Read-only by convention."""
    ball_x: np.ndarray
    ball_y: np.ndarray
    home: np.ndarray        # distance (m) of the home team's nearest player to the ball; inf if none
    away: np.ndarray
    speed: np.ndarray       # ball speed (m/s)
    smooth: np.ndarray      # ... median-smoothed
    changes: np.ndarray     # bool: the ball's path breaks here (rule 11)
    tempo: np.ndarray       # median player speed over 2 s (rules 8, 9)
    width_diff: np.ndarray  # home width minus away width, smoothed (rule 15)
    tags: SmartTags
    left_first_half: int    # team defending the left goal in the first half

    def as_legacy(self):
        """The tuple prepare() used to return: (f dict, kickoffs, corners, goals,
        left_first_half, free_kicks). starter/basic_rules.py unpacks it by position."""
        f = {"ball_x": self.ball_x, "ball_y": self.ball_y, "home": self.home, "away": self.away,
             "speed": self.speed, "smooth": self.smooth, "changes": self.changes,
             "tempo": self.tempo, "width_diff": self.width_diff}
        return (f, self.tags.kickoffs, self.tags.corners, self.tags.goals, self.left_first_half,
                self.tags.free_kicks)


def prepare_match(data_dir, params, ball_path=None):
    """Everything the rules read, once."""
    fps = GAMES[GAME]["fps"]
    features = load_features(data_dir, ball_path)
    suffix = "_keeper" if params.keepers else "_fixed"  # rule 0
    ball_x, ball_y = features["ball_x"], features["ball_y"]
    speed, smooth = ball_kinematics(ball_x, ball_y, fps, params)
    width_diff = pd.Series(features["home_width"] - features["away_width"]).rolling(  # rule 15
        params.shape_smooth, center=True, min_periods=params.shape_smooth // 2).mean().to_numpy()
    meta_path = require(["atd_meta"], GAME, data_dir)[0]
    _, left_first_half = keeper_sides(meta_path)
    return MatchData(
        ball_x=ball_x, ball_y=ball_y,
        home=np.nan_to_num(features["home" + suffix], nan=np.inf),
        away=np.nan_to_num(features["away" + suffix], nan=np.inf),
        speed=speed, smooth=smooth,
        changes=path_changes(ball_x, ball_y, fps, params),  # rule 11
        tempo=tempo(features["player_speed"], params),      # rules 8, 9
        width_diff=width_diff,
        tags=load_tags(data_dir),
        left_first_half=left_first_half)


class HalfContext:
    """One half (video frames first..last inclusive) as the rules see it: the match arrays, the
    settings, and the few arrays every rule derives from them."""

    def __init__(self, match, params, half, first, last):
        self.params = params
        self.half = half
        self.first, self.last = first, last
        self.n = last - first + 1
        self.length, self.width = GAMES[GAME]["pitch"]
        self.fps = GAMES[GAME]["fps"]
        self.left_team = match.left_first_half if half == 1 else opponent(match.left_first_half)
        self.tags = match.tags
        self.x, self.y = match.ball_x, match.ball_y
        self.home, self.away = match.home, match.away
        self.speed, self.smooth = match.speed, match.smooth
        self.changes, self.tempo, self.width_diff = match.changes, match.tempo, match.width_diff
        self.nearest = np.minimum(self.home, self.away)
        self.near_team = np.where(self.home <= self.away, HOME, AWAY)
        self.seen = ~np.isnan(self.x)
        with np.errstate(invalid="ignore"):
            self.kicked = self.seen & (self.speed > params.kick_speed)  # the ball has been kicked
            self.quiet = ~self.seen | (self.smooth < params.quiet_speed)  # not in play (rules 7, 8)

    def first_kick(self, start, search):
        """First frame at or after start (within search frames) where the ball has been kicked."""
        return first_true(self.kicked, start, search)

    def defending(self, x_ball):
        """The team defending the goal on the side of halfway where x_ball is."""
        return self.left_team if x_ball < self.length / 2.0 else opponent(self.left_team)
