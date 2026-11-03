"""Pure numpy helpers: no I/O, no game constants, no rule logic."""

import numpy as np
import pandas as pd

from .config import Params

DEFAULTS = Params()


def tempo(player_speed, params=DEFAULTS):
    """Median player speed, averaged over a centred tempo_frames window; NaN where unknown."""
    speed = player_speed
    known = ~np.isnan(speed)
    total = np.concatenate(([0.0], np.cumsum(np.where(known, speed, 0.0))))
    count = np.concatenate(([0], np.cumsum(known)))
    half = params.tempo_frames // 2
    index = np.arange(len(speed))
    low, high = np.maximum(0, index - half), np.minimum(len(speed), index + half)
    n = count[high] - count[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n >= half, (total[high] - total[low]) / n, np.nan)


def ball_kinematics(x, y, fps, params=DEFAULTS):
    """(speed, smoothed speed) in m/s, NaN where the ball is not seen on both ends."""
    k = 2
    speed = np.full(len(x), np.nan)
    speed[k:-k] = np.hypot(x[2 * k:] - x[:-2 * k], y[2 * k:] - y[:-2 * k]) * fps / (2.0 * k)
    smooth = pd.Series(speed).rolling(params.smooth_frames, center=True,
                                      min_periods=params.smooth_frames // 2).median().to_numpy()
    return speed, smooth


def path_changes(x, y, fps, params=DEFAULTS):
    """True where the ball's path turns or changes speed (rule 11), from path_span frames each side."""
    k = params.path_span
    n = len(x)
    vx_in = np.full(n, np.nan); vy_in = np.full(n, np.nan)
    vx_out = np.full(n, np.nan); vy_out = np.full(n, np.nan)
    vx_in[k:] = x[k:] - x[:-k]; vy_in[k:] = y[k:] - y[:-k]
    vx_out[:-k] = x[k:] - x[:-k]; vy_out[:-k] = y[k:] - y[:-k]
    s_in = np.hypot(vx_in, vy_in) * fps / k
    s_out = np.hypot(vx_out, vy_out) * fps / k
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = (vx_in * vx_out + vy_in * vy_out) / (np.hypot(vx_in, vy_in) * np.hypot(vx_out, vy_out))
        turn = (np.degrees(np.arccos(np.clip(cosine, -1, 1))) >= params.turn_degrees) & (
            np.minimum(s_in, s_out) >= params.min_path_speed)
        ratio = np.maximum(s_in, s_out) / np.minimum(s_in, s_out)
        speed_change = (ratio >= params.speed_ratio) & (np.maximum(s_in, s_out) >= params.kick_speed)
    return np.nan_to_num(turn, nan=0).astype(bool) | np.nan_to_num(speed_change, nan=0).astype(bool)


def runs_of(mask):
    """(starts, ends) of the runs of True in mask; ends are exclusive."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def first_true(mask, start, length):
    """Index of the first True in mask[start:start + length], or None."""
    hits = np.flatnonzero(mask[start:start + length])
    return start + int(hits[0]) if len(hits) else None


def segment_bounds(labels):
    """Indices where a run of equal labels starts, plus 0 and len(labels) at the two ends."""
    return np.concatenate(([0], np.flatnonzero(labels[1:] != labels[:-1]) + 1, [len(labels)]))
