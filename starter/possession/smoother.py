"""Rule 17: a learned smoother over the rules' labels -- a three-state (H, A, D) hidden Markov model.

Per frame, a multinomial logistic regression turns the rules' label and a few tracking signals
into P(H), P(A), P(D) (features() below). Divided by the class shares, these are the emission
scores of an HMM whose transitions cost `switch_team` (H <-> A) and `switch_dead` (into or out
of D) in log space; Viterbi then picks the most likely label path. A short flicker of the rules
is overruled when the signals around it do not back it, and a long ball gap is decided by the
team shape and the rules' labels on both sides of it.

The coefficients are fitted on the DFL open matches only (tools/fit_smoother.py; CC BY 4.0, DFL /
Sportec, Bassek et al. 2025) and stored as JSON. Nothing here was fitted on BAR-ATH labels.
Pure numpy, so the output does not depend on a library version.
"""

import json
import os

import numpy as np
import pandas as pd

CLASSES = ("H", "A", "D")
DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smoother_dfl.json")

FEATURES = ("rule_H", "rule_A", "share_H_2s", "share_A_2s", "share_D_2s", "share_H_10s",
            "share_A_10s", "share_D_10s", "width_diff", "width_known", "seen", "tempo",
            "ball_speed", "near_diff", "ball_toward_away_goal", "ball_to_line", "since_seen")


def _centred_share(flags, window):
    return pd.Series(flags.astype(float)).rolling(window, center=True, min_periods=1).mean().to_numpy()


def features(ctx, labels):
    """(n, len(FEATURES)) float matrix for frames ctx.first..ctx.last; labels: the rules' labels
    for those frames. No NaN in the output."""
    sl = slice(ctx.first, ctx.last + 1)
    fps = ctx.fps
    is_h, is_a, is_d = labels == "H", labels == "A", labels == "D"
    seen = ctx.seen[sl]
    x = pd.Series(ctx.x[sl]).ffill().bfill().fillna(ctx.length / 2).to_numpy()
    y = pd.Series(ctx.y[sl]).ffill().bfill().fillna(ctx.width / 2).to_numpy()
    # +1 at the goal the home team attacks, -1 at its own
    home_attacks_right = ctx.left_team == 0  # HOME == 0 in baseline.py
    toward = (x - ctx.length / 2) / (ctx.length / 2) * (1 if home_attacks_right else -1)
    to_line = np.minimum.reduce([x, ctx.length - x, y, ctx.width - y])
    idx = np.arange(len(seen))
    last_seen = np.maximum.accumulate(np.where(seen, idx, -1))
    since = np.where(last_seen >= 0, idx - last_seen, len(seen))
    with np.errstate(invalid="ignore"):
        near = np.clip(ctx.away[sl] - ctx.home[sl], -10, 10)
    near = np.where(seen & np.isfinite(near), near, 0.0)
    wd = ctx.width_diff[sl]
    tempo = ctx.tempo[sl]
    speed = np.clip(np.nan_to_num(ctx.smooth[sl], nan=0.0), 0, 30)
    cols = [is_h, is_a,
            _centred_share(is_h, 2 * fps + 1), _centred_share(is_a, 2 * fps + 1),
            _centred_share(is_d, 2 * fps + 1),
            _centred_share(is_h, 10 * fps + 1), _centred_share(is_a, 10 * fps + 1),
            _centred_share(is_d, 10 * fps + 1),
            np.clip(np.nan_to_num(wd, nan=0.0), -15, 15), np.isfinite(wd),
            seen, np.clip(np.nan_to_num(tempo, nan=1.5), 0, 6), speed, near, toward,
            np.clip(to_line, 0, 15), np.log1p(np.minimum(since, 30 * fps))]
    return np.column_stack([np.asarray(c, dtype=float) for c in cols])


# -- the model -------------------------------------------------------------------------------------

def fit(X, y, l2=1e-3, iters=300, lr=0.5):
    """Multinomial logistic regression by gradient descent on standardised features.
    y: int class index (0 H, 1 A, 2 D). Returns a JSON-able dict."""
    mean, std = X.mean(axis=0), X.std(axis=0)
    std[std == 0] = 1.0
    Z = np.column_stack([(X - mean) / std, np.ones(len(X))])
    k = len(CLASSES)
    W = np.zeros((Z.shape[1], k))
    Y = np.eye(k)[y]
    for _ in range(iters):
        P = _softmax(Z @ W)
        grad = Z.T @ (P - Y) / len(Z) + l2 * np.vstack([W[:-1], np.zeros((1, k))])
        W -= lr * grad
    prior = np.bincount(y, minlength=k) / float(len(y))
    return {"features": list(FEATURES), "mean": mean.tolist(), "std": std.tolist(),
            "weights": W.tolist(), "prior": prior.tolist()}


def _softmax(S):
    S = S - S.max(axis=1, keepdims=True)
    E = np.exp(S)
    return E / E.sum(axis=1, keepdims=True)


def predict_log_proba(model, X):
    Z = np.column_stack([(X - np.array(model["mean"])) / np.array(model["std"]), np.ones(len(X))])
    return np.log(np.clip(_softmax(Z @ np.array(model["weights"])), 1e-9, 1.0))


def load_model(path=None):
    with open(path or DEFAULT_MODEL, encoding="utf-8") as handle:
        model = json.load(handle)
    if tuple(model["features"]) != FEATURES:
        raise ValueError("smoother model {} was fitted on other features".format(path))
    return model


def viterbi_batch(emissions, switches):
    """Most likely state paths, one per (emission, switch_team, switch_dead) setting in one pass.

    emissions: (G, n, 3) or (n, 3) log scores; switches: G pairs of log transition costs.
    Returns (G, n) int8 paths."""
    switches = np.asarray(switches, dtype=float).reshape(-1, 2)
    g = len(switches)
    emissions = np.broadcast_to(emissions, (g,) + np.shape(emissions)[-2:])
    n = emissions.shape[1]
    team, dead = switches[:, 0], switches[:, 1]
    cost = np.zeros((g, 3, 3))
    cost[:, 0, 1] = cost[:, 1, 0] = team
    cost[:, :2, 2] = cost[:, 2, :2] = dead[:, None]
    back = np.zeros((n, g, 3), dtype=np.int8)
    score = emissions[:, 0].copy()
    rows = np.arange(g)[:, None]
    cols = np.arange(3)[None, :]
    for t in range(1, n):
        cand = score[:, :, None] - cost          # (g, from, to)
        b = np.argmax(cand, axis=1)
        back[t] = b
        score = cand[rows, b, cols] + emissions[:, t]
    paths = np.empty((g, n), dtype=np.int8)
    paths[:, -1] = np.argmax(score, axis=1)
    for t in range(n - 1, 0, -1):
        paths[:, t - 1] = back[t, np.arange(g), paths[:, t]]
    return paths


def viterbi(emission, switch_team, switch_dead):
    """Most likely state path for log-emission scores (n, 3) with transition costs in log space."""
    return viterbi_batch(emission, [(switch_team, switch_dead)])[0]


_MODELS = {}


def smooth(ctx, labels, kick, report):
    """Rule 17 on one half's labels (in place): frames from the kickoff on are re-decided."""
    p = ctx.params
    if not p.smoother:
        return labels
    path = p.smoother_model or DEFAULT_MODEL
    if path not in _MODELS:
        _MODELS[path] = load_model(path)
    model = _MODELS[path]
    start = max(kick - ctx.first, 0)
    if start >= len(labels) - 1:
        return labels
    logp = predict_log_proba(model, features(ctx, labels))[start:]
    emission = logp - np.log(np.array(model["prior"])) * p.smoother_prior
    path_idx = viterbi(emission, p.switch_team, p.switch_dead)
    new = np.array(CLASSES, dtype=labels.dtype)[path_idx]
    report.set("smoother changed frames", int((labels[start:] != new).sum()))
    labels[start:] = new
    return labels
