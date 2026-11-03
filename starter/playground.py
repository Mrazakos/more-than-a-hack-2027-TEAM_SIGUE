"""Playground: the per-frame feature table, the team's rules on any match, and tuning them on
external data.

    python starter/playground.py table [--raw N] [--save]   # 0) the feature table of BAR-ATH
    python starter/playground.py selfcheck                  # 1) our path == best_so_far.py?
    python starter/playground.py dfl-download               # 3) fetch + cache the 7 DFL matches
    python starter/playground.py cv [--n-configs 40]        # 3) leave-one-match-out on DFL
    python starter/playground.py final [--score-public]     # 4) the tuned rules on BAR-ATH

(On Windows: `py -3` instead of `python`.) Without a command it runs `table`.

Rules (organisers' clarification of 2026-11-03):
  - tuning and training only on external data -- here the 7 DFL matches (CC BY 4.0);
  - no ground truth for fitting, tuning or picking: `cv` never touches BAR-ATH at all;
  - do not paste the BAR-ATH table, its rows or its printout into an AI chat -- that is uploading
    the match data. Your code and the DFL numbers are fine;
  - everything written goes to the gitignored data/ folder or experiments/ (parameters and DFL
    scores only, no BAR-ATH data): never force-add data/ to git.
"""

# =================================================================================================
#  ATTEKINTES (a csapatnak)
# -------------------------------------------------------------------------------------------------
#  Ez a fajl negy reszbol all, mindegyik egy-egy blokk lent, nagy fejlec-kommenttel:
#
#  0. RESZ  FEATURE-TABLA      -- frame-enkenti tabla a BAR-ATH meccsrol, nezelodeshez.
#  1. RESZ  A CSAPAT SZABALYAI -- a best_so_far.py szabalymotorja (starter/possession/), de ugy
#                                 becsomagolva, hogy BARMILYEN meccsen fusson, ne csak a BAR-ATH-on.
#  2. RESZ  HIANYZO LABDA      -- uj szabaly: a labda nelkuli frame-ek potlasa
#                                 (a) Kalman-simitoval (fizikai palya a ket ismert pont kozott),
#                                 (b) a jatekosok mozgasabol (merre fut a tobbseg).
#  3. RESZ  KERESZTVALIDACIO   -- 7 nemet Bundesliga-meccs (DFL/IDSSE nyilt adat). Mindig 6 meccsen
#                                 valasztjuk ki a legjobb parametereket, a 7.-en merjuk, es ezt
#                                 mind a 7 meccsel eljatsszuk (leave-one-match-out).
#  4. RESZ  VEGSO FUTTATAS     -- a 7 meccsen legjobb beallitas lefut a BAR-ATH-on, es egy leadhato
#                                 CSV-t ir, amit ellenoriz is.
#
#  Fontos: itt a "tanitas" = parameterek kivalasztasa (random search), nem neuralis halo. A
#  szabalyok ugyanazok, csak a szamokat (pl. 1,5 m erintesi tavolsag) nem a BAR-ATH-on, hanem a
#  nemet meccseken allitjuk be. Ezt keri az uj szabaly.
# =================================================================================================

import argparse
import dataclasses
import json
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from games import DATA_DIR, GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from baseline import (AWAY, FIELD_SEP, GROUP_SEP, HOME, MISSING, NOBODY, RESULTS_DIR,  # noqa: E402
                      team_of_each_track, write_submission)
from possession.config import Params  # noqa: E402
from possession.context import HalfContext, MatchData, prepare_match  # noqa: E402
from possession.labeller import label_half, label_match  # noqa: E402
from possession.report import Report  # noqa: E402
from possession.signals import ball_kinematics, path_changes, runs_of, tempo  # noqa: E402
from possession.tags import SmartTags, load_tags  # noqa: E402
from possession.tracking import MAX_PLAYER_SPEED, keeper_sides, load_features  # noqa: E402
import load_smart_tagging  # noqa: E402
import possession_metrics as pm  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)

FPS = GAMES[GAME]["fps"]  # 25, the DFL data too
EXPERIMENTS = os.path.join(ROOT, "experiments")
BEST_CONFIG = os.path.join(EXPERIMENTS, "playground_best.json")
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


# =================================================================================================
#  0. RESZ -- FEATURE-TABLA (BAR-ATH)
# -------------------------------------------------------------------------------------------------
#  Frame-enkent egy sor. A best_so_far.py-val kozos, gyorsitotarazott feature-okbol indul
#  (possession/tracking.py), es hozzatesz nehany szarmaztatott oszlopot meg a Smart Tagging
#  intervallumait. A sajat otleteidet a my_experiments()-be ird. Itt nincs ground truth.
# =================================================================================================

def build_table(data_dir=None):
    """The per-frame feature table as a DataFrame (first run reads the ATD, a minute or two)."""
    features = load_features(data_dir)
    n = len(features["ball_x"])
    df = pd.DataFrame({name: values for name, values in features.items()})
    df.insert(0, "frame", np.arange(n))

    # which half, and whether the frame is graded at all
    df.insert(1, "half", 0)
    for half, first, last in half_video_blocks(GAME):
        df.loc[first:last, "half"] = half
    df.insert(2, "graded", df["half"] > 0)

    # the ball
    x, y = df["ball_x"].to_numpy(), df["ball_y"].to_numpy()
    df["ball_seen"] = np.isfinite(x)
    df["ball_speed"], df["ball_speed_smooth"] = ball_kinematics(x, y, FPS)
    df["ball_path_change"] = path_changes(x, y, FPS)

    # the players
    df["tempo"] = tempo(df["player_speed"].to_numpy())
    df["width_diff"] = df["home_width"] - df["away_width"]          # > 0: home spread wider
    df["dist_gap"] = df["home_keeper"] - df["away_keeper"]          # < 0: a home player is nearer
    df["nearest_team"] = np.where(df["dist_gap"] < 0, "H", np.where(df["dist_gap"] > 0, "A", ""))
    df["nearest_dist"] = df[["home_keeper", "away_keeper"]].min(axis=1)

    # Smart Tagging: the interval each frame falls in (later tags win where they overlap)
    df["tag_code"] = ""
    df["tag_team"] = ""
    df["tag_type"] = ""
    for event in load_smart_tagging.load(GAME, data_dir):
        lo, hi = max(event["start_frame"], 0), min(event["end_frame"], n)
        if lo >= hi:
            continue
        df.loc[lo:hi - 1, "tag_code"] = event["code"] or ""
        df.loc[lo:hi - 1, "tag_team"] = event["tags"].get("Team") or ""
        if event["tags"].get("Type"):
            df.loc[lo:hi - 1, "tag_type"] = event["tags"]["Type"]
    return df


def load_raw_atd(limit, data_dir=None):
    """The raw ATD through kloppy, every track slot (x, y, speed), for the first `limit` frames."""
    from kloppy import metrica
    from games import asset_paths
    paths = asset_paths(data_dir=data_dir)
    dataset = metrica.load_tracking_epts(meta_data=paths["atd_meta"], raw_data=paths["atd_raw"],
                                         limit=limit)
    return dataset.to_df()


def summary(df):
    graded = df[df["graded"]]
    print("\n{} rows x {} columns ({} graded frames)\n".format(len(df), df.shape[1], len(graded)))
    info = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_% (graded)": (graded.isna().mean() * 100).round(1),
    })
    print(info.to_string())
    print("\nnumeric columns on graded frames:")
    print(graded.describe().T.round(2).to_string())
    print("\nSmart Tagging codes over graded frames (frames):")
    print(graded["tag_code"].replace("", "(none)").value_counts().to_string())


def my_experiments(df):
    """Your playground. df has one row per frame; filter df[df.graded] for match frames."""
    g = df[df["graded"]]
    print("\n--- my_experiments ---")
    print("ball seen on {:.1f}% of graded frames".format(100 * g["ball_seen"].mean()))
    print("someone within 1.5 m of the ball: {:.1f}% of frames with the ball".format(
        100 * (g.loc[g["ball_seen"], "nearest_dist"] <= 1.5).mean()))
    # e.g. how wide each team plays when it is the one nearest the ball:
    print(g.groupby("nearest_team")[["home_width", "away_width"]].mean().round(2).to_string())


# =================================================================================================
#  1. RESZ -- A CSAPAT SZABALYAI, BARMILYEN MECCSEN
# -------------------------------------------------------------------------------------------------
#  A best_so_far.py szabalyai (kapusok, kezdorugas, erintes + visszadatumozas, pass-by, kirugas,
#  szoglet, kapusrugas, gol, szabadrugas, megallasok, alakzat-veto ...) valtozatlanul futnak:
#  a possession.labeller.label_half()-et hivjuk. Ket dolgot kellett altalanositani:
#
#   - features_from_positions(): ugyanazokat a frame-enkenti feature-oket szamolja (legkozelebbi
#     hazai/vendeg jatekos tavolsaga, median jatekossebesseg, csapatszelesseg), mint a
#     possession/tracking.py, csak nem az ATD szovegfajlbol, hanem jatekos-pozicio tombokbol.
#     Igy ugyanaz a kod eszi a BAR-ATH-ot es a nemet meccseket.
#   - run_team_rules(): a felidok hatarait, a palyameretet es az fps-t parameterkent kapja
#     (az eredeti a BAR-ATH konstansait olvassa a games.py-bol).
#
#  `selfcheck` paranccsal ellenorizheto, hogy alapbeallitassal, labdapotlas nelkul ez a ut
#  ugyanazt adja a BAR-ATH-on, mint a best_so_far.py. A kimenet csak darabszam, adatot nem mutat.
# =================================================================================================

class AnyHalf(HalfContext):
    """HalfContext with the pitch and fps of any match instead of BAR-ATH's."""

    def __init__(self, match, params, half, first, last, pitch, fps):
        super().__init__(match, params, half, first, last)
        self.length, self.width = pitch
        self.fps = fps


def features_from_positions(px, py, teams, keepers, bx, by, left_first_half, second_half_from,
                            pitch_length, fps=FPS):
    """The arrays possession/tracking.read_features() makes, from (frames x slots) positions in
    metres (NaN = not tracked) and a ball track. teams: HOME/AWAY per slot; keepers: bool per slot."""
    halfway = pitch_length / 2.0
    n, slots = px.shape
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows are expected

        # median player speed between consecutive frames; a jump faster than 12 m/s is an ID switch
        step = np.full((n, slots), np.nan)
        step[1:] = ((px[1:] - px[:-1]) ** 2 + (py[1:] - py[:-1]) ** 2) ** 0.5 * fps
        step[step > MAX_PLAYER_SPEED] = np.nan
        out["player_speed"] = np.nanmedian(step, axis=1)
        del step

        # each team's width: spread across the pitch of its outfield players (5 or more seen)
        for team, name in ((HOME, "home_width"), (AWAY, "away_width")):
            ys = py[:, (teams == team) & ~keepers]
            width = np.nanstd(ys, axis=1)
            width[np.isfinite(ys).sum(axis=1) < 5] = np.nan
            out[name] = width

        # each team's nearest player to the ball, by slot team (_fixed) and with goalkeepers given
        # to the team defending their side of halfway (_keeper, rule 0)
        dist = ((px - bx[:, None]) ** 2 + (py - by[:, None]) ** 2) ** 0.5
        left_team = np.where(np.arange(n) < second_half_from, left_first_half, 1 - left_first_half)
        team_of = np.broadcast_to(teams, (n, slots)).copy()
        out["ball_x"], out["ball_y"] = bx.copy(), by.copy()
        for name, team in (("home", HOME), ("away", AWAY)):
            out[name + "_fixed"] = np.nanmin(np.where(team_of == team, dist, np.nan), axis=1)
        for slot in np.flatnonzero(keepers):
            on_left = px[:, slot] < halfway
            team_of[:, slot] = np.where(on_left, left_team, 1 - left_team)
        for name, team in (("home", HOME), ("away", AWAY)):
            out[name + "_keeper"] = np.nanmin(np.where(team_of == team, dist, np.nan), axis=1)
    return out


def match_from_features(features, tags, left_first_half, params, fps=FPS):
    """MatchData, exactly as possession.context.prepare_match() builds it, from feature arrays."""
    suffix = "_keeper" if params.keepers else "_fixed"  # rule 0
    ball_x, ball_y = features["ball_x"], features["ball_y"]
    speed, smooth = ball_kinematics(ball_x, ball_y, fps, params)
    width_diff = pd.Series(features["home_width"] - features["away_width"]).rolling(  # rule 15
        params.shape_smooth, center=True, min_periods=params.shape_smooth // 2).mean().to_numpy()
    return MatchData(
        ball_x=ball_x, ball_y=ball_y,
        home=np.nan_to_num(features["home" + suffix], nan=np.inf),
        away=np.nan_to_num(features["away" + suffix], nan=np.inf),
        speed=speed, smooth=smooth,
        changes=path_changes(ball_x, ball_y, fps, params),  # rule 11
        tempo=tempo(features["player_speed"], params),      # rules 8, 9
        width_diff=width_diff, tags=tags, left_first_half=left_first_half)


def run_team_rules(match, params, blocks, pitch, fps=FPS):
    """The team's labeller (possession.labeller.label_half) over the given halves: [(half, first,
    last)] inclusive. Frames outside the halves are X."""
    labels = np.full(len(match.ball_x), "X", dtype="<U1")
    for half, first, last in blocks:
        ctx = AnyHalf(match, params, half, first, last, pitch, fps)
        labels[first:last + 1] = label_half(ctx, Report())
    return labels


def read_atd_positions(data_dir=None):
    """BAR-ATH's ATD as arrays: px, py (frames x slots, metres) and bx, by. Cached next to the data."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, data_dir)
    cache = os.path.join(os.path.dirname(raw_path), "playground_positions_v1_{}.npz".format(GAME))
    if os.path.isfile(cache) and os.path.getmtime(cache) >= os.path.getmtime(raw_path):
        with np.load(cache) as stored:
            return {key: stored[key] for key in stored.files}
    length, width = GAMES[GAME]["pitch"]
    n = source_frames("tactical", GAME)
    slots = len(team_of_each_track(meta_path))
    px = np.full((n, slots), np.nan)
    py = np.full((n, slots), np.nan)
    bx = np.full(n, np.nan)
    by = np.full(n, np.nan)
    print("reading {} into position arrays (once; cached afterwards) ...".format(
        os.path.basename(raw_path)))
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1  # frame_id counts from 1, submissions from 0
            if not 0 <= frame < n:
                continue
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if group[0] == MISSING:
                    continue
                values = group.split(",")
                px[frame, slot] = float(values[0]) * length
                py[frame, slot] = float(values[1]) * width
            if parts[2][0] != MISSING:
                atd = parts[2].split(",")
                bx[frame], by[frame] = float(atd[0]) * length, float(atd[1]) * width
    out = {"px": px, "py": py, "bx": bx, "by": by}
    np.savez_compressed(cache, **out)
    return out


def barath_inputs(data_dir=None):
    """Everything the playground needs from BAR-ATH, in the same shape as a DFL match."""
    meta_path = require(["atd_meta"], GAME, data_dir)[0]
    pos = read_atd_positions(data_dir)
    keeper_slots, left_first_half = keeper_sides(meta_path)
    teams = team_of_each_track(meta_path)
    keepers = np.zeros(len(teams), dtype=bool)
    keepers[list(keeper_slots)] = True
    blocks = half_video_blocks(GAME)
    return dict(px=pos["px"], py=pos["py"], bx=pos["bx"], by=pos["by"], teams=teams,
                keepers=keepers, left_first_half=left_first_half, tags=load_tags(data_dir),
                blocks=blocks, second_half_from=(blocks[0][2] + blocks[1][1]) // 2,
                pitch=GAMES[GAME]["pitch"], gt=None)


def selfcheck(data_dir=None):
    """Does the playground's path give best_so_far.py's labels on BAR-ATH? Prints counts only."""
    m = barath_inputs(data_dir)
    mine = match_features(m)
    official = load_features(data_dir)
    print("feature arrays (playground vs possession/tracking.py):")
    for key in sorted(official):
        a, b = official[key], mine[key]
        nan_differs = int((np.isnan(a) != np.isnan(b)).sum())
        both = np.isfinite(a) & np.isfinite(b)
        diff = float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0
        print("  {:<14} NaN pattern differs on {:>6} frames, max abs diff {:.2e}".format(
            key, nan_differs, diff))
    params = Params()
    reference = label_match(prepare_match(data_dir, params), params)
    ours = run_team_rules(match_from_features(mine, m["tags"], m["left_first_half"], params),
                          params, m["blocks"], m["pitch"])
    differ = int((reference != ours).sum())
    print("labels: {} of {} frames differ from best_so_far.py -> {}".format(
        differ, len(ours), "OK, identical" if differ == 0 else "CHECK THIS"))


# =================================================================================================
#  2. RESZ -- HIANYZO LABDA (uj szabaly)
# -------------------------------------------------------------------------------------------------
#  A BAR-ATH-on a labda a pontozott frame-ek 41%-an hianyzik, es a szabalyok ilyenkor vakok:
#  nincs erintes, nincs kirugas-detektalas, a regi csapat "megtartja" a labdat. Ket potlas:
#
#  (a) kalman_fill() -- FIZIKAI INTERPOLACIO. Allando sebessegu modell (pozicio + sebesseg),
#      Kalman-szuro elore, majd Rauch-Tung-Striebel simito visszafele. A labda talajra vetitett
#      palyaja ivelt labdanal is kozel egyenes, egyenletes mozgas, a modell gyorsulas-zaja pedig
#      megengedi a lassulast vagy a kanyart. A simito a res MINDKET vegen ismert poziciohoz es
#      sebesseghez illeszkedik, ezert sima, fizikailag hiheto palyat ad a lyukba. Csak a legfeljebb
#      `kalman_gap` frame hosszu reseket toltjuk ki, mert hosszabb res alatt valaki biztosan
#      hozzaert. A kalman_gap = 0 kikapcsolja.
#
#  (b) players_ball_estimate() -- A JATEKOSOK MOZGASABOL. Ahol a Kalman sem segit, a mozgo
#      jatekosok (>= 2 m/s) futasi iranyabol becsuljuk a labdat: az a pont, amely a futasi
#      iranyokat jelolo egyenesekhez a legkozelebb esik (sulyozott legkisebb negyzetek, zart
#      kepletu 2x2 megoldas). Kicsit a jatekosok sulypontja fele huzzuk, hogy parhuzamos futasnal
#      se szaljon el. A becsles palyan belulre van szoritva, ezert soha nem general hamis "kint
#      a labda" esemenyt. A player_ball = True kapcsolja be.
#
#  Hogy a ket potlas segit-e, azt NEM mi dontjuk el: a kalman_gap es a player_ball is a 3. resz
#  parameterkeresesenek resze, tehat a nemet meccsek mondjak meg.
# =================================================================================================

KALMAN_ACCEL = 10.0        # m/s^2: the process noise -- how hard the ball may brake or bend
KALMAN_MEAS = 0.15         # m: measurement noise of a tracked ball position
PLAYER_MIN_SPEED = 2.0     # m/s: slower players say nothing about where the ball is
PLAYER_MIN_MOVERS = 4      # fewer moving players than this: no estimate
PLAYER_PRIOR_WEIGHT = 2.0  # the players' centroid counts as this many average movers


def kalman_fill(x, y, blocks, max_gap, fps=FPS, accel=KALMAN_ACCEL, meas=KALMAN_MEAS):
    """(x, y, filled) with every gap of at most max_gap frames inside a half filled by a
    constant-velocity Kalman filter + RTS smoother. Seen frames keep their values."""
    x, y = x.copy(), y.copy()
    filled = np.zeros(len(x), dtype=bool)
    if max_gap <= 0:
        return x, y, filled
    dt = 1.0 / fps
    q00, q01, q11 = accel ** 2 * dt ** 4 / 4, accel ** 2 * dt ** 3 / 2, accel ** 2 * dt ** 2
    r = meas ** 2
    for _, first, last in blocks:
        seen = np.flatnonzero(np.isfinite(x[first:last + 1])) + first
        if len(seen) < 2:
            continue
        k0, k1 = seen[0], seen[-1]
        m = k1 - k0 + 1
        # forward: filtered means (fx, fvx, fy, fvy), covariances (shared by x and y: same
        # observations), and the one-step predictions the smoother needs
        fx, fvx, fy, fvy = (np.zeros(m) for _ in range(4))
        px_, pvx, py_, pvy = (np.zeros(m) for _ in range(4))
        f00, f01, f11 = (np.zeros(m) for _ in range(3))
        p00, p01, p11 = (np.zeros(m) for _ in range(3))
        mx, vx, my, vy = x[k0], 0.0, y[k0], 0.0
        c00, c01, c11 = r, 0.0, 100.0
        fx[0], fvx[0], fy[0], fvy[0], f00[0], f01[0], f11[0] = mx, vx, my, vy, c00, c01, c11
        obs_x, obs_y = x[k0:k1 + 1], y[k0:k1 + 1]
        for i in range(1, m):
            mx, my = mx + dt * vx, my + dt * vy
            c00, c01, c11 = c00 + 2 * dt * c01 + dt * dt * c11 + q00, c01 + dt * c11 + q01, c11 + q11
            px_[i], pvx[i], py_[i], pvy[i], p00[i], p01[i], p11[i] = mx, vx, my, vy, c00, c01, c11
            zx = obs_x[i]
            if zx == zx:  # observed (not NaN)
                s = c00 + r
                g0, g1 = c00 / s, c01 / s
                ix, iy = zx - mx, obs_y[i] - my
                mx, vx, my, vy = mx + g0 * ix, vx + g1 * ix, my + g0 * iy, vy + g1 * iy
                c00, c01, c11 = (1 - g0) * c00, (1 - g0) * c01, c11 - g1 * c01
            fx[i], fvx[i], fy[i], fvy[i], f00[i], f01[i], f11[i] = mx, vx, my, vy, c00, c01, c11
        # backward (Rauch-Tung-Striebel): smoothed means
        sx, svx, sy, svy = fx[-1], fvx[-1], fy[-1], fvy[-1]
        smooth_x, smooth_y = np.zeros(m), np.zeros(m)
        smooth_x[-1], smooth_y[-1] = sx, sy
        for i in range(m - 2, -1, -1):
            a, b, c = p00[i + 1], p01[i + 1], p11[i + 1]
            det = a * c - b * b
            # gain C = Pf F^T inv(Pp), with Pf F^T = [[f00 + dt f01, f01], [f01 + dt f11, f11]]
            m00, m01 = f00[i] + dt * f01[i], f01[i]
            m10, m11 = f01[i] + dt * f11[i], f11[i]
            c00_ = (m00 * c - m01 * b) / det
            c01_ = (-m00 * b + m01 * a) / det
            c10_ = (m10 * c - m11 * b) / det
            c11_ = (-m10 * b + m11 * a) / det
            ex, evx = sx - px_[i + 1], svx - pvx[i + 1]
            ey, evy = sy - py_[i + 1], svy - pvy[i + 1]
            sx, svx = fx[i] + c00_ * ex + c01_ * evx, fvx[i] + c10_ * ex + c11_ * evx
            sy, svy = fy[i] + c00_ * ey + c01_ * evy, fvy[i] + c10_ * ey + c11_ * evy
            smooth_x[i], smooth_y[i] = sx, sy
        # fill the short gaps strictly between two seen frames
        starts, ends = runs_of(~np.isfinite(x[k0:k1 + 1]))
        for s0, s1 in zip(starts, ends):
            if s1 - s0 <= max_gap:
                x[k0 + s0:k0 + s1] = smooth_x[s0:s1]
                y[k0 + s0:k0 + s1] = smooth_y[s0:s1]
                filled[k0 + s0:k0 + s1] = True
    return x, y, filled


def players_ball_estimate(px, py, todo, pitch, fps=FPS, min_speed=PLAYER_MIN_SPEED,
                          min_movers=PLAYER_MIN_MOVERS, prior_weight=PLAYER_PRIOR_WEIGHT):
    """(x, y) estimates on the frames in `todo` (NaN where too few players move): the point
    closest to the lines the moving players run along, pulled towards their centroid."""
    n = len(px)
    idx = np.flatnonzero(todo)
    ex, ey = np.full(n, np.nan), np.full(n, np.nan)
    if not len(idx):
        return ex, ey
    k = 2
    lo, hi = np.clip(idx - k, 0, n - 1), np.clip(idx + k, 0, n - 1)
    span = np.maximum(hi - lo, 1)[:, None] / fps
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        vx, vy = (px[hi] - px[lo]) / span, (py[hi] - py[lo]) / span
        speed = np.hypot(vx, vy)
        w = np.where(speed >= min_speed, speed, 0.0)
        w = np.nan_to_num(w)
        ux, uy = np.nan_to_num(vx / speed), np.nan_to_num(vy / speed)
        x0, y0 = np.nan_to_num(px[idx]), np.nan_to_num(py[idx])
        # projector onto the normal of each running line: I - u u^T
        a = (w * (1 - ux * ux)).sum(1)
        b = (w * (-ux * uy)).sum(1)
        c = (w * (1 - uy * uy)).sum(1)
        rx = (w * ((1 - ux * ux) * x0 - ux * uy * y0)).sum(1)
        ry = (w * (-ux * uy * x0 + (1 - uy * uy) * y0)).sum(1)
        movers = (w > 0).sum(1)
        lam = prior_weight * w.sum(1) / np.maximum(movers, 1)
        cx, cy = np.nanmean(px[idx], axis=1), np.nanmean(py[idx], axis=1)
        a, c, rx, ry = a + lam, c + lam, rx + lam * cx, ry + lam * cy
        det = a * c - b * b
        gx, gy = (c * rx - b * ry) / det, (a * ry - b * rx) / det
    ok = (movers >= min_movers) & np.isfinite(gx) & np.isfinite(gy)
    length, width = pitch
    ex[idx[ok]] = np.clip(gx[ok], 0.5, length - 0.5)  # never over a line: no fake exits
    ey[idx[ok]] = np.clip(gy[ok], 0.5, width - 0.5)
    return ex, ey


def fill_ball(m, kalman_gap, player_ball, fps=FPS):
    """The ball track of match m with rule 2a (Kalman) and optionally 2b (players) applied."""
    bx, by, _ = kalman_fill(m["bx"], m["by"], m["blocks"], kalman_gap, fps)
    if player_ball:
        todo = ~np.isfinite(bx)
        inside = np.zeros(len(bx), dtype=bool)
        for _, first, last in m["blocks"]:
            inside[first:last + 1] = True
        ex, ey = players_ball_estimate(m["px"], m["py"], todo & inside, m["pitch"], fps)
        got = np.isfinite(ex)
        bx, by = bx.copy(), by.copy()
        bx[got], by[got] = ex[got], ey[got]
    return bx, by


def match_features(m, kalman_gap=0, player_ball=False):
    """The feature arrays of match m with the chosen ball filling."""
    bx, by = fill_ball(m, kalman_gap, player_ball)
    return features_from_positions(m["px"], m["py"], m["teams"], m["keepers"], bx, by,
                                   m["left_first_half"], m["second_half_from"], m["pitch"][0])


# =================================================================================================
#  3. RESZ -- KULSO ADAT + KERESZTVALIDACIO (leave-one-match-out)
# -------------------------------------------------------------------------------------------------
#  ADAT: 7 Bundesliga-meccs a DFL/IDSSE nyilt adathalmazbol (Bassek et al. 2025, Sci Data, CC BY
#  4.0), a kloppy tolti le (sportec.load_open_*). Kepkockankent benne van a "BallStatus"
#  (el/halott) es a "BallPossession" (melyik csapate), vagyis PONTOSAN az, amibol a szervezok a
#  mi ground truth-unkat is kepzik: halott -> D, egyebkent a birtoklo csapat -> H/A.
#
#  A DFL tracking javitott, tiszta adat, a mienk nem. Ezert degrade_like_atd() elrontja:
#  a labda kb. 41%-ban hianyozzon (ezt a szamot a szervezok DATA.md-je kozli, nem a GT-bol
#  jon), foleg ha a levegoben van, a pozicioknak legyen zaja, es a jatekosok neha tunjenek el.
#  Ezek a szamok (ATD_LIKE) feltetelezesek, a dokumentacioban le kell irni oket.
#
#  A DFL "BallPossession" mar egy-egy erintesre is atvalt (a spellek 17%-a 1 s-nal rovidebb), a
#  kiiras szerint viszont "egy egyszeru erintes kontroll nelkul nem birtokvaltas". Ezert
#  control_filter() a GT_MIN_CONTROL_S-nal rovidebb, a masik csapat birtoklasa utan kozvetlenul
#  kovetkezo spelleket visszaadja az elozo csapatnak. Ez a definiciobol jon, nem a GT-bol.
#  0-ra allitva kikapcsolhato.
#
#  A Smart Tagging helyett a DFL esemenyekbol kepzunk "tag"-eket: kezdorugas, szoglet,
#  szabadrugas, gol utani kezdes. Mivel a Smart Tagging automatikus es pontatlan, a tag-eket
#  TAG_LEAD_S masodperccel korabbra tesszuk, es a szabadrugasoknak csak egy reszet tartjuk meg
#  (a BAR-ATH-on 5 szabadrugas-tag van, egy meccsen altalaban kb. 20 szabadrugas tortenik).
#
#  KERESZTVALIDACIO: az osszes jelolt beallitast (random search a SEARCH_SPACE-bol, a 0. jelolt
#  a csapat mostani alapbeallitasa) lefuttatjuk mind a 7 meccsen. Utana minden "fold"-ban:
#     tanulo = 6 meccs  -> azt a jeloltet valasztjuk, amelyiknek ezen a legjobb az atlag spellF1-e
#     teszt  = 1 meccs  -> ezen merjuk a valasztott jeloltet (es osszehasonlitaskent a 0.-t)
#  Mind a 7 meccs pontosan egyszer teszt. A 7 teszteredmeny atlaga becsli, mennyit er a hangolas
#  egy SOHA NEM LATOTT meccsen. A vegleges beallitas az, ami mind a 7-en legjobb. Ez kerul az
#  experiments/playground_best.json-ba, es ezt hasznalja a 4. resz.
# =================================================================================================

DFL_MATCHES = ("J03WMX", "J03WN1", "J03WPY", "J03WOH", "J03WQQ", "J03WOY", "J03WR9")
HALFTIME_GAP = 250  # X frames put between the two halves of a DFL match

ATD_LIKE = {                 # how the DFL tracking is made to look like the automatic ATD
    "ball_missing": 0.41,    # share of frames without a ball (DATA.md: 41% of graded frames)
    "ball_gap_s": 1.0,       # mean length of a ball-missing stretch, seconds (assumed)
    "air_height": 2.0,       # m: a ball higher than this ...
    "air_missing": 0.8,      # ... is lost for its whole flight with this probability (assumed)
    "ball_noise": 0.2,       # m: position noise of the ball (assumed)
    "player_missing": 0.07,  # share of frames a player is not tracked (DATA.md: 20.5 of 22 seen)
    "player_gap_s": 2.0,     # mean length of such a stretch, seconds (assumed)
    "player_noise": 0.3,     # m: position noise of the players (assumed)
    "noise_rho": 0.99,       # frame-to-frame correlation of the noise (a wobble, not a jitter)
}
GT_MIN_CONTROL_S = 1.0       # s: a shorter DFL possession right after the other team's is a
                             # touch without control (CHALLENGE.md); 0 keeps DFL's own labels
TAG_LEAD_S = 2.0             # s: tags open this much before the event (Smart Tagging is early)
FREE_KICK_TAG_KEEP = 0.25    # share of free kicks that get a tag

SEARCH_SPACE = {             # the values the random search draws from (rules 2, 11, 9, 8, 13-15, 2a/b)
    "touch_radius": [0.8, 1.0, 1.25, 1.5, 1.75, 2.0],
    "backdate_max": [0, 25, 50, 75, 100],
    "control_speed": [2.0, 3.0, 4.0],
    "dead_lead": [0, 5, 10, 15, 20, 25],
    "dead_tail": [0, 8, 17, 25],
    "shape_veto": [0.0, 4.0, 6.0, 8.0],
    "merge_gap": [0.0, 3.0, 5.0, 8.0],
    "slow_tempo": [1.0, 1.2, 1.5],
    "kalman_gap": [0, 12, 25, 50],
    "player_ball": [False, True],
}
BALL_KEYS = ("kalman_gap", "player_ball")


def dfl_dir(data_dir=None):
    return os.path.join(data_dir or DATA_DIR, "external", "dfl")


def dfl_extract(match_id, data_dir=None):
    """Download one DFL match with kloppy and cache it as arrays (a few minutes, once)."""
    path = os.path.join(dfl_dir(data_dir), "{}.npz".format(match_id))
    if os.path.isfile(path):
        with np.load(path) as stored:
            return {key: stored[key] for key in stored.files}
    from kloppy import sportec
    from kloppy.domain import BallState, Ground, SetPieceQualifier, SetPieceType

    started = time.time()
    print("  {}: downloading + reading tracking (kloppy) ...".format(match_id), flush=True)
    tracking = sportec.load_open_tracking_data(match_id=match_id)
    meta = tracking.metadata
    length, width = meta.pitch_dimensions.pitch_length, meta.pitch_dimensions.pitch_width
    players = [p for team in meta.teams for p in team.players]
    slot_of = {p.player_id: i for i, p in enumerate(players)}
    home_team = next(t for t in meta.teams if t.ground == Ground.HOME)
    teams = np.array([HOME if p.team == home_team else AWAY for p in players], dtype=np.int8)
    keepers = np.array([any("Goalkeeper" in str(v) for v in p.positions.items.values())
                        or "Goalkeeper" in str(p.starting_position) for p in players])

    frames = tracking.frames
    period = np.array([f.period.id for f in frames])
    frame_id = np.array([f.frame_id for f in frames])
    n1, n2 = int((period == 1).sum()), int((period == 2).sum())
    start2 = n1 + HALFTIME_GAP
    first_id = {1: frame_id[period == 1].min(), 2: frame_id[period == 2].min()}
    n = start2 + n2
    px = np.full((n, len(players)), np.nan, dtype=np.float32)
    py = np.full((n, len(players)), np.nan, dtype=np.float32)
    bx, by, bz = (np.full(n, np.nan, dtype=np.float32) for _ in range(3))
    alive = np.zeros(n, dtype=bool)
    owner = np.full(n, NOBODY, dtype=np.int8)
    for f in frames:
        if f.period.id not in (1, 2):
            continue
        i = (0 if f.period.id == 1 else start2) + f.frame_id - first_id[f.period.id]
        b = f.ball_coordinates
        if b is not None and b.x is not None:
            bx[i], by[i], bz[i] = b.x * length, b.y * width, (b.z or 0.0)
        alive[i] = f.ball_state == BallState.ALIVE
        if f.ball_owning_team is not None:
            owner[i] = HOME if f.ball_owning_team == home_team else AWAY
        for player, data in f.players_data.items():
            if data.coordinates is not None and player.player_id in slot_of:
                s = slot_of[player.player_id]
                px[i, s], py[i, s] = data.coordinates.x * length, data.coordinates.y * width

    # which team defends the left goal in the first half: its goalkeeper stands there
    gk_x = [np.nanmean(px[:n1, s]) for s in np.flatnonzero(keepers & (teams == HOME))]
    left_first_half = HOME if gk_x and np.nanmean(gk_x) < length / 2 else AWAY

    print("  {}: reading events ...".format(match_id), flush=True)
    events = sportec.load_open_event_data(match_id=match_id)
    kickoffs, corners, free_kicks = [], [], []
    for e in events.events:
        kind = e.get_qualifier_value(SetPieceQualifier)
        if kind is None or e.period.id not in (1, 2):
            continue
        i = (0 if e.period.id == 1 else start2) + int(round(e.timestamp.total_seconds() * FPS))
        team = HOME if e.team is not None and e.team.ground == Ground.HOME else AWAY
        if kind == SetPieceType.KICK_OFF:
            kickoffs.append((i, e.period.id))
        elif kind == SetPieceType.CORNER_KICK:
            corners.append((i, team))
        elif kind == SetPieceType.FREE_KICK:
            free_kicks.append((i, team))
    out = dict(px=px, py=py, bx=bx, by=by, bz=bz, alive=alive, owner=owner, teams=teams,
               keepers=keepers, left_first_half=np.int8(left_first_half),
               halves=np.array([[1, 0, n1 - 1], [2, start2, n - 1]]),
               pitch=np.array([length, width]),
               kickoffs=np.array(kickoffs, dtype=np.int64).reshape(-1, 2),
               corners=np.array(corners, dtype=np.int64).reshape(-1, 2),
               free_kicks=np.array(free_kicks, dtype=np.int64).reshape(-1, 2))
    os.makedirs(dfl_dir(data_dir), exist_ok=True)
    np.savez_compressed(path, **out)
    print("  {}: {} frames, {} players, cached in {:.0f} s".format(
        match_id, n, len(players), time.time() - started), flush=True)
    return out


def markov_mask(n, share, mean_gap, rng):
    """Bool array: True in stretches of mean length mean_gap frames, covering about share of n."""
    out = np.zeros(n, dtype=bool)
    if share <= 0:
        return out
    mean_on = mean_gap * (1 - share) / share
    state, pos = rng.random() < share, 0
    while pos < n:
        run = rng.geometric(1.0 / (mean_gap if state else mean_on))
        if state:
            out[pos:pos + run] = True
        pos += run
        state = not state
    return out


def ar_noise(shape, sigma, rho, rng):
    """Correlated noise (AR(1)) along axis 0: a slow wobble of standard deviation sigma."""
    noise = np.empty(shape)
    noise[0] = rng.normal(0, sigma, shape[1:])
    step = sigma * np.sqrt(1 - rho ** 2)
    shocks = rng.normal(0, step, shape)
    for t in range(1, shape[0]):
        noise[t] = rho * noise[t - 1] + shocks[t]
    return noise


def degrade_like_atd(raw, rng, c=ATD_LIKE, fps=FPS):
    """px, py, bx, by of a clean DFL match, made to look like the automatic ATD."""
    n, slots = raw["px"].shape
    px = raw["px"].astype(float) + ar_noise((n, slots), c["player_noise"], c["noise_rho"], rng)
    py = raw["py"].astype(float) + ar_noise((n, slots), c["player_noise"], c["noise_rho"], rng)
    for s in range(slots):
        gone = markov_mask(n, c["player_missing"], c["player_gap_s"] * fps, rng)
        px[gone, s] = np.nan
        py[gone, s] = np.nan
    ball_noise = ar_noise((n, 2), c["ball_noise"], c["noise_rho"], rng)
    bx = raw["bx"].astype(float) + ball_noise[:, 0]
    by = raw["by"].astype(float) + ball_noise[:, 1]
    # a ball in the air is lost for its whole flight with probability air_missing ...
    in_air = np.nan_to_num(raw["bz"]) > c["air_height"]
    lost = np.zeros(n, dtype=bool)
    for a, b in zip(*runs_of(in_air)):
        if rng.random() < c["air_missing"]:
            lost[a:b] = True
    # ... and the rest of the 41% in stretches of ball_gap_s seconds
    rest = max(0.0, 1 - (1 - c["ball_missing"]) / max(1e-9, 1 - lost.mean()))
    lost |= markov_mask(n, min(rest, 0.95), c["ball_gap_s"] * fps, rng)
    bx[lost] = np.nan
    by[lost] = np.nan
    return px, py, bx, by


def control_filter(gt, min_frames):
    """In place: a possession shorter than min_frames that directly follows the other team's is
    a touch without control (CHALLENGE.md), so the previous team keeps the ball."""
    if min_frames <= 0:
        return gt
    for start, stop in pm.valid_runs(gt):
        segments = pm.segments(gt, [(start, stop)])
        for (_, a1, _), (b0, b1, here) in zip(segments, segments[1:]):
            before = gt[a1 - 1]  # as already filtered
            if (here in (pm.H, pm.A) and before in (pm.H, pm.A) and before != here
                    and b1 - b0 < min_frames):
                gt[b0:b1] = before
    return gt


def dfl_match(match_id, data_dir=None, seed=0):
    """One DFL match in the playground's shape: degraded tracking, tags, halves, ground truth."""
    raw = dfl_extract(match_id, data_dir)
    rng = np.random.default_rng([seed, DFL_MATCHES.index(match_id)])
    px, py, bx, by = degrade_like_atd(raw, rng)
    n = len(bx)
    blocks = [tuple(int(v) for v in row) for row in raw["halves"]]
    lead = int(round(TAG_LEAD_S * FPS))

    kickoffs, goals = {}, []
    for frame, half in raw["kickoffs"]:
        if int(half) not in kickoffs:
            kickoffs[int(half)] = int(frame)          # the first one of a half starts it
        else:
            goals.append(int(frame) - lead)           # the others follow a goal
    corners = [(int(f) - lead, int(t)) for f, t in raw["corners"]]
    free_kicks = [(int(f) - lead, int(t)) for f, t in raw["free_kicks"]
                  if rng.random() < FREE_KICK_TAG_KEEP]
    tags = SmartTags(kickoffs, sorted(corners), sorted(goals), sorted(free_kicks))

    gt = np.full(n, pm.X, dtype=np.int8)                  # the labels, as the organisers make them
    for _, first, last in blocks:
        part = slice(first, last + 1)
        gt[part] = np.where(~raw["alive"][part], pm.D,
                            np.where(raw["owner"][part] == HOME, pm.H, pm.A))
    control_filter(gt, int(round(GT_MIN_CONTROL_S * FPS)))
    return dict(px=px, py=py, bx=bx, by=by, teams=raw["teams"], keepers=raw["keepers"],
                left_first_half=int(raw["left_first_half"]), tags=tags, blocks=blocks,
                second_half_from=blocks[1][1] - HALFTIME_GAP // 2,
                pitch=tuple(float(v) for v in raw["pitch"]), gt=gt)


def codes_of(labels):
    """H/A/D/X letters -> the scorer's codes."""
    codes = np.full(len(labels), pm.X, dtype=np.int8)
    for name, code in (("H", pm.H), ("A", pm.A), ("D", pm.D)):
        codes[labels == name] = code
    return codes


def score(gt_codes, labels):
    """(spellF1, macroF1) with the official definitions (evaluation/possession_metrics.py)."""
    pred = codes_of(labels)
    runs = pm.valid_runs(gt_codes)
    spell = pm.spell_scores(gt_codes, pm.fill_abstentions(pred, runs), runs)["mean_f1"]
    macro = pm.macro_f1(pm.scored_confusion(pm.confusion(gt_codes, pred)))
    return spell, macro


def make_candidates(n_configs, seed):
    """Candidate settings: 0 = the team's defaults, 1 = defaults + Kalman, then random draws."""
    defaults = Params()
    base = {k: getattr(defaults, k) for k in SEARCH_SPACE if k not in BALL_KEYS}
    base.update(kalman_gap=0, player_ball=False)
    candidates = [base, dict(base, kalman_gap=25)]
    rng = np.random.default_rng(seed)
    seen = {json.dumps(c, sort_keys=True) for c in candidates}
    while len(candidates) < n_configs:
        draw = {k: values[rng.integers(len(values))] for k, values in SEARCH_SPACE.items()}
        draw = {k: (v.item() if hasattr(v, "item") else v) for k, v in draw.items()}
        key = json.dumps(draw, sort_keys=True)
        if key not in seen:
            seen.add(key)
            candidates.append(draw)
    return candidates


def params_of(config):
    """(Params, kalman_gap, player_ball) of one candidate."""
    rule_settings = {k: v for k, v in config.items() if k not in BALL_KEYS}
    return (dataclasses.replace(Params(), **rule_settings),
            int(config.get("kalman_gap", 0)), bool(config.get("player_ball", False)))


def label_with(m, config, feature_cache=None):
    """Labels of match m under one candidate (features cached per ball-filling choice)."""
    params, kalman_gap, player_ball = params_of(config)
    key = (kalman_gap, player_ball)
    if feature_cache is None or key not in feature_cache:
        features = match_features(m, kalman_gap, player_ball)
        if feature_cache is not None:
            feature_cache[key] = features
    else:
        features = feature_cache[key]
    match = match_from_features(features, m["tags"], m["left_first_half"], params)
    return run_team_rules(match, params, m["blocks"], m["pitch"])


def cross_validate(n_configs=40, seed=0, data_dir=None):
    candidates = make_candidates(n_configs, seed)
    spell = np.zeros((len(candidates), len(DFL_MATCHES)))
    macro = np.zeros_like(spell)

    print("scoring {} candidates on {} DFL matches ...".format(len(candidates), len(DFL_MATCHES)))
    for j, match_id in enumerate(DFL_MATCHES):
        started = time.time()
        m = dfl_match(match_id, data_dir, seed)
        cache = {}
        for i, config in enumerate(candidates):
            spell[i, j], macro[i, j] = score(m["gt"], label_with(m, config, cache))
        print("  {}  defaults spellF1 {:.3f}   best {:.3f}   ({:.0f} s)".format(
            match_id, spell[0, j], spell[:, j].max(), time.time() - started), flush=True)

    # leave-one-match-out: pick on 6, test on the 7th, every match once
    print("\nleave-one-match-out")
    print("  test match  picked  train spellF1  TEST spellF1  test macroF1  | defaults on test")
    rows = []
    for j, match_id in enumerate(DFL_MATCHES):
        train = [k for k in range(len(DFL_MATCHES)) if k != j]
        mean_spell, mean_macro = spell[:, train].mean(1), macro[:, train].mean(1)
        best = int(np.lexsort((-mean_macro, -mean_spell))[0])  # spellF1, then macroF1
        rows.append(dict(test=match_id, picked=best, train_spell=mean_spell[best],
                         test_spell=spell[best, j], test_macro=macro[best, j],
                         defaults_test_spell=spell[0, j]))
        print("  {:<10}  #{:<5}  {:>13.3f}  {:>12.3f}  {:>12.3f}  | {:.3f}".format(
            match_id, best, mean_spell[best], spell[best, j], macro[best, j], spell[0, j]))
    test = np.array([r["test_spell"] for r in rows])
    dflt = np.array([r["defaults_test_spell"] for r in rows])
    print("\n  tuned:    TEST spellF1 {:.3f} +- {:.3f}   (what to expect on an unseen match)".format(
        test.mean(), test.std()))
    print("  defaults: TEST spellF1 {:.3f} +- {:.3f}   (the team's settings, untuned)".format(
        dflt.mean(), dflt.std()))

    final = int(np.lexsort((-macro.mean(1), -spell.mean(1)))[0])
    print("\nfinal choice (best mean over all 7): candidate #{}".format(final))
    for key, value in candidates[final].items():
        print("  {:<14} {}".format(key, value))

    os.makedirs(EXPERIMENTS, exist_ok=True)
    with open(BEST_CONFIG, "w") as handle:
        json.dump({"config": candidates[final], "candidate": final, "n_configs": n_configs,
                   "seed": seed, "atd_like": ATD_LIKE, "gt_min_control_s": GT_MIN_CONTROL_S,
                   "tag_lead_s": TAG_LEAD_S,
                   "free_kick_tag_keep": FREE_KICK_TAG_KEEP,
                   "cv_test_spell_mean": float(test.mean()), "cv_test_spell_std": float(test.std()),
                   "cv_defaults_spell_mean": float(dflt.mean()),
                   "folds": [{k: (float(v) if isinstance(v, float) else v) for k, v in r.items()}
                             for r in rows]}, handle, indent=2)
    table = pd.DataFrame(spell, columns=DFL_MATCHES)
    table.insert(0, "candidate", range(len(candidates)))
    table["config"] = [json.dumps(c, sort_keys=True) for c in candidates]
    table.to_csv(os.path.join(EXPERIMENTS, "playground_cv_scores.csv"), index=False)
    print("\nwrote {} and playground_cv_scores.csv (DFL scores and settings only)".format(
        os.path.relpath(BEST_CONFIG, ROOT)))


# =================================================================================================
#  4. RESZ -- VEGSO FUTTATAS A BAR-ATH-ON
# -------------------------------------------------------------------------------------------------
#  Beolvassa a 3. resz altal kivalasztott beallitast (experiments/playground_best.json), egyszer
#  lefuttatja vele a szabalyokat a BAR-ATH-on (labdapotlassal, ha a CV azt valasztotta), kiirja az
#  evaluation/results/playground.csv-t, es a hivatalos validatorral ellenorzi (sorszam, cimkek).
#
#  --score-public: a nyilvanos 5 perces GT-n is kiirja a pontszamot. FIGYELEM: ez csak riport.
#  Ha a szam alapjan barmit visszaallitasz a beallitasokon, az mar GT-n hangolas, ami tilos.
#  Hogy a puszta meres megengedett-e, azt erdemes megkerdezni a szervezoktol. Addig csak akkor
#  hasznald, ha tudod, hogy semmit nem fogsz miatta valtoztatni.
# =================================================================================================

def run_final(data_dir=None, config_path=BEST_CONFIG, score_public=False):
    if not os.path.isfile(config_path):
        sys.exit("{} not found: run `playground.py cv` first".format(config_path))
    with open(config_path) as handle:
        config = json.load(handle)["config"]
    print("settings from {}:".format(os.path.relpath(config_path, ROOT)))
    for key, value in config.items():
        print("  {:<14} {}".format(key, value))

    m = barath_inputs(data_dir)
    labels = label_with(m, config)
    path = write_submission(os.path.join(RESULTS_DIR, "playground.csv"), labels)
    problem = pm.validate_submission(pm.read_label_csv(path), source_frames("tactical", GAME))
    print("\nwrote {}: {}".format(os.path.relpath(path, ROOT), problem or "valid submission"))

    if score_public:
        print("\n[report only -- do not tune on this] public 5 min:")
        gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
        spell, macro = score(gt, labels)
        print("  spellF1 {:.4f}   macroF1 {:.4f}".format(spell, macro))


# =================================================================================================
#  5. RESZ -- TANULT KONTROLL-MODELL + VITERBI (a kezi kuszobok helyett)
# -------------------------------------------------------------------------------------------------
#  Az 1-4. resz kezzel irt szabalyai helyett itt egy tanult modell dont:
#
#  (a) ml_features(): frame-enkent kb. 60 jellemzo. Labda: lathato-e, mioha nem latszik,
#      sebesseg, iranytores, hol van, kint van-e. Csapatonkent: legkozelebbi es masodik
#      legkozelebbi jatekos tavolsaga, hanyan vannak 3 m-en es 8 m-en belul (nyomas), a
#      legkozelebbi jatekos sebessege es kozeledesi sebessege, a csapat sulypontja es annak
#      elorehaladasa a sajat tamadasi iranyaban, szelesseg es hosszusag. Kozos: tempo. Ezek
#      gordulo atlagai a frame elott, utan es korulotte (kontextus). Tag-ek: mennyi ido telt el
#      az utolso kezdorugas, szoglet, szabadrugas vagy gol ota, es mennyi van a kovetkezoig.
#
#  (b) LightGBM (gradient boosting dontesi fakkal): minden frame-re P(H), P(A), P(D). A DFL
#      meccseken tanul, mindegyiket TUKROZVE is (hazai <-> vendeg csere), igy a modell nem tanulja
#      meg, hogy "a hazai csapatnal szokott lenni a labda", csak azt, hogy hogyan nez ki a birtoklas.
#
#  (c) Viterbi-dekodolas (rejtett Markov-modell, HMM): a frame-enkenti valoszinusegekbol a
#      legvaloszinubb H/A/D SOROZATOT keresi. Minden csapatvaltas `switch`, minden halott-labda
#      ki- es belepes `dead` buntetopontba kerul. Igy egy-ket zajos frame nem csinal uj birtoklast,
#      de egy tartos, egyertelmu valtas igen. Ez egyben valtja ki a visszadatumozast, az
#      osszevonast, a dead_tail-t es az alakzat-vetot.
#
#  KIERTEKELES -- egymasba agyazott (nested) leave-one-match-out:
#     kulso kor: 1 meccs a teszt, 6 a tanulo.
#     belso kor: a 6 tanulobol mindig 1-et felreteszunk, a maradek 5-on tanitunk, a felretetten
#                kiprobaljuk az osszes (switch, dead) buntetest. A 6 felretett meccsen atlagosan
#                legjobb buntetes nyer.
#     Utana a modellt mind a 6 tanulon tanitjuk, es a nyertes buntetessel MERJUK a tesztmeccset.
#  Igy a tesztmeccs semmilyen dontesbe nem szol bele, ezert a szam becsuletes becsles egy soha
#  nem latott meccsre. Osszehasonlitaskent kiirjuk a Viterbi nelkuli (frame-enkenti argmax)
#  eredmenyt, es ha van, a szabaly-alapu CV eredmenyet is.
#
#  `ml-final`: a modellt mind a 7 meccsen tanitja (fix seed, determinisztikus), a buntetest a
#  CV-bol veszi (experiments/playground_ml.json), es megirja az evaluation/results/playground_ml.csv
#  fajlt a BAR-ATH-ra. A --score-public itt is csak riport.
# =================================================================================================

ML_JSON = os.path.join(EXPERIMENTS, "playground_ml.json")
ML_KALMAN_GAP = 25        # rule 2a, as the rule-based CV chose it
ML_TRAIN_STEP = 5         # train on every 5th frame (neighbours are nearly identical anyway)
ML_SWITCH = [4.0, 8.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0]  # Viterbi penalty for H <-> A
ML_DEAD = [8.0, 16.0, 32.0, 48.0, 64.0, 96.0, 128.0]      # ... for entering or leaving D
ML_ROUNDS = 250           # boosting rounds (trees per class)
ML_PARAMS = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=63,
                 min_data_in_leaf=200, bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8,
                 lambda_l2=1.0, seed=0, deterministic=True, force_row_wise=True, verbose=-1)
STATE_LABELS = np.array(["H", "A", "D"])


def _clock(events, n, cap=1500):
    """(frames since the last event at or before t, frames until the next one), capped."""
    events = np.sort(np.asarray(events, dtype=np.int64))
    t = np.arange(n)
    since = np.full(n, cap, dtype=np.float32)
    until = np.full(n, cap, dtype=np.float32)
    if len(events):
        i = np.searchsorted(events, t, side="right")
        has_prev, has_next = i > 0, i < len(events)
        since[has_prev] = np.minimum(t[has_prev] - events[i[has_prev] - 1], cap)
        until[has_next] = np.minimum(events[i[has_next]] - t[has_next], cap)
    return since, until


def _rolling(values, window, how="center"):
    s = pd.Series(values)
    if how == "center":
        return s.rolling(window, center=True, min_periods=1).mean().to_numpy(np.float32)
    if how == "past":
        return s.rolling(window, min_periods=1).mean().to_numpy(np.float32)
    return s[::-1].rolling(window, min_periods=1).mean()[::-1].to_numpy(np.float32)  # future


def ml_features(m, flip=False, kalman_gap=ML_KALMAN_GAP, fps=FPS):
    """(frames x features float32, names). flip swaps the two teams (for training both ways)."""
    teams = (1 - m["teams"]) if flip else m["teams"]
    left = (1 - m["left_first_half"]) if flip else m["left_first_half"]
    length, width = m["pitch"]
    n = len(m["bx"])
    f = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        # -- the ball ------------------------------------------------------------------------
        seen = np.isfinite(m["bx"])
        bx, by, filled = kalman_fill(m["bx"], m["by"], m["blocks"], kalman_gap, fps)
        f["ball_seen"], f["ball_filled"] = seen, filled
        frames_seen = np.flatnonzero(seen)
        f["since_seen"], f["until_seen"] = _clock(frames_seen, n, cap=250)
        speed, smooth = ball_kinematics(bx, by, fps)
        f["ball_speed"], f["ball_speed_smooth"] = speed, smooth
        f["path_change"] = pd.Series(path_changes(bx, by, fps).astype(float)).rolling(
            7, center=True, min_periods=1).max().to_numpy()
        left_team = np.where(np.arange(n) < m["second_half_from"], left, 1 - left)
        home_right = left_team == HOME  # home defends the left goal, so it attacks to the right
        f["ball_x_home_attack"] = np.where(home_right, bx, length - bx)
        f["ball_y_centre"] = np.abs(by - width / 2)
        f["ball_out"] = (bx < 0) | (bx > length) | (by < 0) | (by > width)
        f["ball_line_dist"] = np.minimum.reduce([bx, length - bx, by, width - by])

        # -- the players, per team -----------------------------------------------------------
        px, py = m["px"].astype(np.float32), m["py"].astype(np.float32)
        slots = px.shape[1]
        team_of = np.broadcast_to(teams, (n, slots)).copy()
        for slot in np.flatnonzero(m["keepers"]):  # rule 0: a keeper plays for his goal's team
            team_of[:, slot] = np.where(px[:, slot] < length / 2, left_team, 1 - left_team)
        k = 2
        vx, vy = np.full_like(px, np.nan), np.full_like(py, np.nan)
        vx[k:-k] = (px[2 * k:] - px[:-2 * k]) * fps / (2 * k)
        vy[k:-k] = (py[2 * k:] - py[:-2 * k]) * fps / (2 * k)
        dx, dy = bx[:, None] - px, by[:, None] - py
        dist = np.hypot(dx, dy)
        rows = np.arange(n)
        for team, tag in ((HOME, "h"), (AWAY, "a")):
            mine = team_of == team
            d = np.where(mine & np.isfinite(dist), dist, np.inf)
            two = np.argpartition(d, 1, axis=1)[:, :2]
            d2 = np.sort(np.take_along_axis(d, two, axis=1), axis=1)
            nearest = np.where(d[rows, two[:, 0]] <= d[rows, two[:, 1]], two[:, 0], two[:, 1])
            f[tag + "_d1"] = np.where(np.isinf(d2[:, 0]), np.nan, d2[:, 0])
            f[tag + "_d2"] = np.where(np.isinf(d2[:, 1]), np.nan, d2[:, 1])
            f[tag + "_n3"] = (d < 3).sum(1)
            f[tag + "_n8"] = (d < 8).sum(1)
            known = np.isfinite(f[tag + "_d1"])  # no ball or nobody of the team: no nearest player
            nvx = np.where(known, vx[rows, nearest], np.nan)
            nvy = np.where(known, vy[rows, nearest], np.nan)
            f[tag + "_near_speed"] = np.hypot(nvx, nvy)
            f[tag + "_closing"] = (nvx * dx[rows, nearest] + nvy * dy[rows, nearest]) / np.maximum(
                dist[rows, nearest], 0.1)
            outfield_x = np.where(mine, px, np.nan)
            outfield_y = np.where(mine, py, np.nan)
            cx = np.nanmean(outfield_x, axis=1)
            attacks_right = left_team == team
            cx_attack = np.where(attacks_right, cx, length - cx)
            f[tag + "_centroid_attack"] = cx_attack
            push = np.full(n, np.nan)
            push[12:-12] = (cx_attack[24:] - cx_attack[:-24]) * fps / 24
            f[tag + "_push"] = push
            f[tag + "_width"] = np.nanstd(outfield_y, axis=1)
            f[tag + "_length"] = np.nanstd(outfield_x, axis=1)
            f[tag + "_seen"] = np.isfinite(outfield_x).sum(1)
        step = np.full((n, slots), np.nan, dtype=np.float32)
        step[1:] = np.hypot(px[1:] - px[:-1], py[1:] - py[:-1]) * fps
        step[step > MAX_PLAYER_SPEED] = np.nan
        f["tempo"] = _rolling(np.nanmedian(step, axis=1), 50)
        del step, dist, dx, dy, vx, vy

        # -- context: what happens around the frame ------------------------------------------
        diff = np.clip(np.nan_to_num(f["a_d1"] - f["h_d1"], nan=0.0), -15, 15)  # > 0: home nearer
        near = np.sign(diff)
        f["d1_diff"] = diff
        for window in (25, 101):
            f["near_c{}".format(window)] = _rolling(near, window)
            f["diff_c{}".format(window)] = _rolling(diff, window)
        f["near_past25"] = _rolling(near, 25, "past")
        f["near_future25"] = _rolling(near, 25, "future")
        f["near_past100"] = _rolling(near, 100, "past")
        f["near_future100"] = _rolling(near, 100, "future")
        f["speed_c101"] = _rolling(smooth, 101)
        f["out_c51"] = _rolling(f["ball_out"].astype(float), 51)
        f["seen_c51"] = _rolling(seen.astype(float), 51)
        f["width_diff_c51"] = _rolling(f["h_width"] - f["a_width"], 51)
        f["push_diff_c51"] = _rolling(f["h_push"] - f["a_push"], 51)

        # -- Smart Tagging (or the DFL stand-in) ---------------------------------------------
        tags = m["tags"]
        sign = -1.0 if flip else 1.0
        f["kickoff_since"], f["kickoff_until"] = _clock(list(tags.kickoffs.values()), n)
        f["goal_since"], f["goal_until"] = _clock(tags.goals, n)
        for name, events in (("corner", tags.corners), ("fk", tags.free_kicks)):
            frames = [e[0] for e in events]
            f[name + "_since"], f[name + "_until"] = _clock(frames, n)
            side = np.zeros(n, dtype=np.float32)  # +1 home's, -1 away's, for the last such tag
            for frame, team in sorted(events):
                if 0 <= frame < n and team in (HOME, AWAY):
                    side[frame:] = sign * (1.0 if team == HOME else -1.0)
            f[name + "_team"] = side

    names = list(f)
    matrix = np.column_stack([np.asarray(f[k], dtype=np.float32) for k in names])
    matrix[np.isinf(matrix)] = np.nan
    return matrix, names


def ml_match_data(m):
    """Full features for prediction plus mirrored, subsampled training rows."""
    X, names = ml_features(m)
    inside = np.zeros(len(X), dtype=bool)
    for _, first, last in m["blocks"]:
        inside[first:last + 1] = True
    data = dict(X=X, names=names, blocks=m["blocks"], gt=m.get("gt"))
    if m.get("gt") is not None:
        rows = np.flatnonzero(inside)[::ML_TRAIN_STEP]
        Xf, _ = ml_features(m, flip=True)
        y = m["gt"][rows].astype(np.int8)          # 0 H, 1 A, 2 D
        y_flip = np.where(y == pm.H, pm.A, np.where(y == pm.A, pm.H, y))
        data["X_train"] = np.vstack([X[rows], Xf[rows]])
        data["y_train"] = np.concatenate([y, y_flip])
        del Xf
    return data


def ml_fit(datas):
    import lightgbm
    X = np.vstack([d["X_train"] for d in datas])
    y = np.concatenate([d["y_train"] for d in datas])
    model = lightgbm.train(ML_PARAMS, lightgbm.Dataset(X, y, feature_name=datas[0]["names"]),
                           num_boost_round=ML_ROUNDS)
    prior = np.bincount(y, minlength=3) / float(len(y))
    return model, prior


def ml_emissions(model, prior, data):
    """log P(state | frame) - log P(state): the HMM's scaled likelihoods."""
    proba = model.predict(data["X"])
    return np.log(np.clip(proba, 1e-6, 1.0)) - np.log(prior)[None, :]


def viterbi(logp, switch, dead):
    """The best H/A/D state path (0/1/2) with `switch` per H<->A change and `dead` per D entry or exit."""
    n = len(logp)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    lp = logp.tolist()
    s0, s1, s2 = lp[0]
    back = bytearray(3 * n)
    for t in range(1, n):
        e0, e1, e2 = lp[t]
        a, b, c = s0, s1 - switch, s2 - dead   # into H
        if a >= b and a >= c:
            n0, p0 = a, 0
        elif b >= c:
            n0, p0 = b, 1
        else:
            n0, p0 = c, 2
        a, b, c = s0 - switch, s1, s2 - dead   # into A
        if b >= a and b >= c:
            n1, p1 = b, 1
        elif a >= c:
            n1, p1 = a, 0
        else:
            n1, p1 = c, 2
        a, b, c = s0 - dead, s1 - dead, s2     # into D
        if c >= a and c >= b:
            n2, p2 = c, 2
        elif a >= b:
            n2, p2 = a, 0
        else:
            n2, p2 = b, 1
        back[3 * t], back[3 * t + 1], back[3 * t + 2] = p0, p1, p2
        s0, s1, s2 = n0 + e0, n1 + e1, n2 + e2
    path = np.empty(n, dtype=np.int8)
    state = int(np.argmax([s0, s1, s2]))
    for t in range(n - 1, -1, -1):
        path[t] = state
        state = back[3 * t + state]
    return path


def ml_decode(logp, blocks, switch, dead):
    """H/A/D letters over the halves (X outside); switch=None: plain per-frame argmax."""
    labels = np.full(len(logp), "X", dtype="<U1")
    for _, first, last in blocks:
        part = logp[first:last + 1]
        path = part.argmax(1) if switch is None else viterbi(part, switch, dead)
        labels[first:last + 1] = STATE_LABELS[path]
    return labels


def team_acc(gt_codes, labels):
    return pm.team_accuracy(pm.scored_confusion(pm.confusion(gt_codes, codes_of(labels))))[1]


def ml_cross_validate(data_dir=None, seed=0):
    grid = [(s, d) for s in ML_SWITCH for d in ML_DEAD]
    started = time.time()
    data = {}
    print("features for the {} DFL matches ...".format(len(DFL_MATCHES)), flush=True)
    for match_id in DFL_MATCHES:
        data[match_id] = ml_match_data(dfl_match(match_id, data_dir, seed))
    print("  done ({:.0f} s), {} features".format(time.time() - started,
                                                 len(data[DFL_MATCHES[0]]["names"])), flush=True)

    rows, outer_logp = [], {}
    for j in DFL_MATCHES:
        train = [k for k in DFL_MATCHES if k != j]
        inner = np.zeros(len(grid))
        for k in train:  # inner loop: pick the Viterbi penalties without ever seeing j
            model, prior = ml_fit([data[i] for i in train if i != k])
            logp = ml_emissions(model, prior, data[k])
            for g, (sw, sd) in enumerate(grid):
                inner[g] += score(data[k]["gt"], ml_decode(logp, data[k]["blocks"], sw, sd))[0]
        sw, sd = grid[int(np.argmax(inner))]
        model, prior = ml_fit([data[i] for i in train])
        logp = ml_emissions(model, prior, data[j])
        outer_logp[j] = logp
        gt = data[j]["gt"]
        labels = ml_decode(logp, data[j]["blocks"], sw, sd)
        spell, macro = score(gt, labels)
        raw_spell, raw_macro = score(gt, ml_decode(logp, data[j]["blocks"], None, None))
        rows.append(dict(test=j, switch=sw, dead=sd, test_spell=spell, test_macro=macro,
                         test_team_acc=team_acc(gt, labels), argmax_spell=raw_spell,
                         argmax_macro=raw_macro))
        print("  test {}  switch {:>4} dead {:>4}  spellF1 {:.3f}  macroF1 {:.3f}  team acc {:.3f}"
              "  | argmax only: spellF1 {:.3f}  ({:.0f} s)".format(
                  j, sw, sd, spell, macro, rows[-1]["test_team_acc"], raw_spell,
                  time.time() - started), flush=True)

    test = np.array([r["test_spell"] for r in rows])
    print("\n  ML + Viterbi:  TEST spellF1 {:.3f} +- {:.3f}   macroF1 {:.3f}   team acc {:.3f}".format(
        test.mean(), test.std(), np.mean([r["test_macro"] for r in rows]),
        np.mean([r["test_team_acc"] for r in rows])))
    print("  ML argmax:     TEST spellF1 {:.3f}   (the same model without Viterbi)".format(
        np.mean([r["argmax_spell"] for r in rows])))
    if os.path.isfile(BEST_CONFIG):
        with open(BEST_CONFIG) as handle:
            rules = json.load(handle)
        print("  rules (cv):    TEST spellF1 {:.3f}   team defaults {:.3f}".format(
            rules["cv_test_spell_mean"], rules["cv_defaults_spell_mean"]))

    # the penalties for the final model: best mean over the 7 out-of-match predictions
    totals = np.zeros(len(grid))
    for j in DFL_MATCHES:
        for g, (sw, sd) in enumerate(grid):
            totals[g] += score(data[j]["gt"], ml_decode(outer_logp[j], data[j]["blocks"], sw, sd))[0]
    sw, sd = grid[int(np.argmax(totals))]
    print("\nfinal Viterbi penalties: switch {}  dead {}".format(sw, sd))
    os.makedirs(EXPERIMENTS, exist_ok=True)
    with open(ML_JSON, "w") as handle:
        json.dump({"switch": sw, "dead": sd, "kalman_gap": ML_KALMAN_GAP, "seed": seed,
                   "train_step": ML_TRAIN_STEP, "lgbm": ML_PARAMS, "lgbm_rounds": ML_ROUNDS,
                   "atd_like": ATD_LIKE,
                   "gt_min_control_s": GT_MIN_CONTROL_S, "tag_lead_s": TAG_LEAD_S,
                   "free_kick_tag_keep": FREE_KICK_TAG_KEEP,
                   "features": data[DFL_MATCHES[0]]["names"],
                   "cv_test_spell_mean": float(test.mean()), "cv_test_spell_std": float(test.std()),
                   "folds": [{k: (float(v) if isinstance(v, float) else v) for k, v in r.items()}
                             for r in rows]}, handle, indent=2)
    print("wrote {} (DFL scores and settings only)".format(os.path.relpath(ML_JSON, ROOT)))


def ml_run_final(data_dir=None, score_public=False):
    if not os.path.isfile(ML_JSON):
        sys.exit("{} not found: run `playground.py ml-cv` first".format(ML_JSON))
    with open(ML_JSON) as handle:
        setup = json.load(handle)
    print("training on all {} DFL matches (switch {}, dead {}) ...".format(
        len(DFL_MATCHES), setup["switch"], setup["dead"]), flush=True)
    model, prior = ml_fit([ml_match_data(dfl_match(k, data_dir, setup["seed"]))
                           for k in DFL_MATCHES])
    m = barath_inputs(data_dir)
    data = ml_match_data(m)
    labels = ml_decode(ml_emissions(model, prior, data), m["blocks"], setup["switch"], setup["dead"])
    path = write_submission(os.path.join(RESULTS_DIR, "playground_ml.csv"), labels)
    problem = pm.validate_submission(pm.read_label_csv(path), source_frames("tactical", GAME))
    print("wrote {}: {}".format(os.path.relpath(path, ROOT), problem or "valid submission"))
    if score_public:
        print("\n[report only -- do not tune on this] public 5 min:")
        spell, macro = score(pm.label_codes(pm.read_label_csv(PUBLIC_GT)), labels)
        print("  spellF1 {:.4f}   macroF1 {:.4f}".format(spell, macro))


# =================================================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=None, help="data bundle, if not ./data or MTAH_DATA_DIR")
    sub = parser.add_subparsers(dest="cmd")
    table = sub.add_parser("table", help="0) the BAR-ATH feature table")
    table.add_argument("--raw", type=int, default=0, metavar="N",
                       help="also load and describe the raw ATD for the first N frames (kloppy)")
    table.add_argument("--save", action="store_true",
                       help="write the table to data/playground_features.csv (gitignored)")
    sub.add_parser("selfcheck", help="1) the playground path reproduces best_so_far.py")
    sub.add_parser("dfl-download", help="3) download and cache the 7 DFL matches")
    cv = sub.add_parser("cv", help="3) leave-one-match-out tuning on the DFL matches")
    cv.add_argument("--n-configs", type=int, default=40, help="candidates in the random search")
    cv.add_argument("--seed", type=int, default=0)
    final = sub.add_parser("final", help="4) the tuned rules on BAR-ATH -> playground.csv")
    final.add_argument("--config", default=BEST_CONFIG)
    final.add_argument("--score-public", action="store_true",
                       help="also print the public-window score (report only, never tune on it)")
    ml_cv = sub.add_parser("ml-cv", help="5) learned model + Viterbi, nested leave-one-match-out")
    ml_cv.add_argument("--seed", type=int, default=0)
    ml_final = sub.add_parser("ml-final", help="5) the learned model on BAR-ATH -> playground_ml.csv")
    ml_final.add_argument("--score-public", action="store_true",
                          help="also print the public-window score (report only, never tune on it)")
    args = parser.parse_args()
    cmd = args.cmd or "table"

    if cmd == "table":
        df = build_table(args.data_dir)
        summary(df)
        my_experiments(df)
        if getattr(args, "raw", 0):
            raw = load_raw_atd(args.raw, args.data_dir)
            print("\nraw ATD: {} rows x {} columns; first columns: {}".format(
                len(raw), raw.shape[1], list(raw.columns[:12])))
        if getattr(args, "save", False):
            out = os.path.join(args.data_dir or DATA_DIR, "playground_features.csv")
            df.to_csv(out, index=False)
            print("\nwrote {}".format(out))
    elif cmd == "selfcheck":
        selfcheck(args.data_dir)
    elif cmd == "dfl-download":
        for match_id in DFL_MATCHES:
            dfl_extract(match_id, args.data_dir)
    elif cmd == "cv":
        cross_validate(args.n_configs, args.seed, args.data_dir)
    elif cmd == "final":
        run_final(args.data_dir, args.config, args.score_public)
    elif cmd == "ml-cv":
        ml_cross_validate(args.data_dir, args.seed)
    elif cmd == "ml-final":
        ml_run_final(args.data_dir, args.score_public)


if __name__ == "__main__":
    main()
