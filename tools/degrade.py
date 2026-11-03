"""Make external tracking look like the ATD: where the ball is lost, and for how long.

DFL's optical tracking has the ball on every frame; the ATD loses it on about 41% of graded frames.
The mask here is drawn from the ATD's own run lengths (visible runs and missing runs, inside the
two halves) -- ATD tracking only, no labels and no video. Seeded, so two parameter sets are scored
on identical degraded input.

Two masks:
  * ball_mask (the first one): alternating runs drawn at random from the ATD's run lengths. It
    assumes the gaps are not tied to play -- they are (see the context mask below).
  * context_mask (the default for tuning): the ATD's missing rate by tempo and pitch zone.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "starter")):
    if p not in sys.path:
        sys.path.insert(0, p)


def runs(flags):
    """Lengths of the consecutive True / False runs of a bool array: ([True runs], [False runs])."""
    edges = np.flatnonzero(flags[1:] != flags[:-1]) + 1
    starts = np.concatenate(([0], edges))
    lengths = np.diff(np.concatenate((starts, [len(flags)])))
    values = flags[starts]
    return lengths[values], lengths[~values]


def atd_ball_run_pools(data_dir=None):
    """(visible run lengths, missing run lengths, missing share) of the ATD ball inside the halves."""
    from games import GAME, half_video_blocks
    from possession.tracking import load_features
    seen = ~np.isnan(load_features(data_dir)["ball_x"])
    visible, missing, total, lost = [], [], 0, 0
    for _half, first, last in half_video_blocks(GAME):
        span = seen[first:last + 1]
        v, m = runs(span)
        visible.append(v)
        missing.append(m)
        total += len(span)
        lost += int((~span).sum())
    return np.concatenate(visible), np.concatenate(missing), lost / total


def ball_mask(blocks, n_frames, pools, seed):
    """Bool per frame, True where the ball is 'seen': alternating runs sampled from the ATD pools."""
    visible, missing, _share = pools
    rng = np.random.default_rng(seed)
    mask = np.zeros(n_frames, dtype=bool)
    for _half, first, last in blocks:
        t, state = first, bool(rng.random() < 0.5)
        while t <= last:
            length = int(rng.choice(visible if state else missing))
            mask[t:min(t + length, last + 1)] = state
            t += length
            state = not state
    return mask


# -- context mask ----------------------------------------------------------------------------------
# The ATD does not lose the ball at random: it is missing on ~78% of frames while the players walk
# (tempo < 1 m/s) but on ~10% in fast play, and on ~70% within 10 m of a goal line. The context mask
# keeps that: P(missing | tempo bin, end zone) is read from the ATD (tracking only), smoothed over
# SMOOTH frames, and thresholded against a random level held for ~SEGMENT frames. Checked on the
# ATD's own tempo and ball: run lengths (median visible 318 vs 345 frames, missing 256 vs 268) and
# the number of gaps (173 vs 150) come out close to the real ones.

TEMPO_EDGES = (1.0, 1.5, 2.0, 3.0)  # m/s: five tempo bins
END_ZONE = 10.0                      # m from a goal line
SMOOTH = 250                         # frames the missing probability is averaged over
SEGMENT = 250                        # frames: mean length of a held random level


def context_bins(tempo_values, ball_x, length):
    """Bin per frame: tempo bin * 2 + (1 if the ball is within END_ZONE of a goal line)."""
    t = np.digitize(np.nan_to_num(tempo_values, nan=1.0), TEMPO_EDGES)
    x = np.asarray(ball_x, dtype=float)
    end = (x < END_ZONE) | (x > length - END_ZONE)
    return t * 2 + end.astype(int)


def atd_context_table(data_dir=None):
    """P(ball missing) per context bin (10 values), from the ATD's tracking only."""
    import pandas as pd
    from games import GAME, GAMES, half_video_blocks
    from possession.config import Params
    from possession.signals import tempo
    from possession.tracking import load_features
    f = load_features(data_dir)
    seen = ~np.isnan(f["ball_x"])
    last_x = pd.Series(f["ball_x"]).ffill().bfill().to_numpy()  # where the ball was last seen
    bins = context_bins(tempo(f["player_speed"], Params()), last_x, GAMES[GAME]["pitch"][0])
    idx = np.concatenate([np.arange(a, b + 1) for _h, a, b in half_video_blocks(GAME)])
    overall = float(np.mean(~seen[idx]))
    table = np.full(2 * (len(TEMPO_EDGES) + 1), overall)
    for k in range(len(table)):
        in_bin = idx[bins[idx] == k]
        if len(in_bin) >= 500:
            table[k] = float(np.mean(~seen[in_bin]))
    return table


def context_mask(blocks, tempo_values, ball_x, length, table, seed):
    """Bool per frame, True where the ball is 'seen', missing where the ATD would likely miss it."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    p = table[context_bins(tempo_values, ball_x, length)]
    mask = np.zeros(len(p), dtype=bool)
    for _half, first, last in blocks:
        span = pd.Series(p[first:last + 1]).rolling(SMOOTH, center=True, min_periods=1).mean().to_numpy()
        level = np.empty(len(span))
        t = 0
        while t < len(span):
            n = int(rng.geometric(1.0 / SEGMENT))
            level[t:t + n] = rng.random()
            t += n
        mask[first:last + 1] = level >= span
    return mask


if __name__ == "__main__":
    visible, missing, share = atd_ball_run_pools()
    print("ATD ball missing on {:.1f}% of frames inside the halves; {} visible runs (median {} frames), "
          "{} missing runs (median {} frames)".format(100 * share, len(visible), int(np.median(visible)),
                                                      len(missing), int(np.median(missing))))
    table = atd_context_table()
    print("P(missing) by tempo bin (rows, edges {}) x [middle, end zone]:".format(TEMPO_EDGES))
    print(np.round(table.reshape(-1, 2), 3))
