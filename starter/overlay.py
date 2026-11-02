"""Draw the tracking on top of the tactical video, to see what the possession rules saw.

For each requested frame, saves an image of the video frame with:

  - every tracked player, as a ring at his feet in his tracked team's colour, with his track slot
    number -- H blue, A red; the player nearest the ball gets a thicker ring
  - the ATD ball (yellow), and your detector's ball boxes if --detections is given (white)
  - the pitch outline and halfway line projected through the homography: if they do not sit on
    the painted lines, nothing else in the image can be trusted
  - a header with the ground truth (public window only), your submission's label, and the
    distance from the ball to the nearest H and A player

    python starter/overlay.py 11351
    python starter/overlay.py 11351,11708,15646 --submission evaluation/results/possession_v3.csv
    python starter/overlay.py 11340:11480:10 --detections data/tactical/BAR-ATH_ball_detections.csv
    python starter/overlay.py 11340:11480 --mp4                   # every frame, as a clip
    python starter/overlay.py 11351 --weights models/player_detector.pt   # + your model's boxes

What to look for at a "wrong team" error: is the ring nearest the ball the right colour for the
shirt under it? A red ring on a Barcelona shirt is a tracking team error, and the fix is in the
tracking, not in the possession rules.

Looking at the video to understand errors is fine; writing labels from it into a submission is
not (RULES.md). The images contain match video: they go to the gitignored data folder and must
not be committed, shared or uploaded anywhere.

    pip install opencv-python
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, GAMES, require  # noqa: E402
from baseline import AWAY, FIELD_SEP, GROUP_SEP, HOME, MISSING, team_of_each_track  # noqa: E402
import load_homography  # noqa: E402

# Params
GT_PATH = os.path.join(ROOT, "evaluation", "gt", "{}_possession_gt_public.csv")
OUT_FOLDER = "overlay"
# Colours are BGR, as OpenCV wants them.
TEAM_COLOURS = {HOME: (255, 128, 0), AWAY: (40, 40, 230)}
BALL_COLOUR = (0, 230, 255)
DETECTION_COLOUR = (255, 255, 255)
PITCH_COLOUR = (80, 255, 80)
MODEL_COLOUR = (200, 200, 200)
RING_RADIUS_M = 1.0           # ring drawn around each player's feet, in metres on the pitch
MP4_FPS = 25


def parse_frames(spec):
    """'11351' | '11351,11708' | '11340:11480' | '11340:11480:10' -> sorted list of frames."""
    frames = set()
    for part in spec.split(","):
        part = part.strip()
        if ":" in part:
            pieces = [int(p) for p in part.split(":")]
            start, end = pieces[0], pieces[1]
            step = pieces[2] if len(pieces) > 2 else 1
            frames.update(range(start, end + 1, step))
        elif part:
            frames.add(int(part))
    return sorted(frames)


def read_tracking(raw_path, frames):
    """{video frame: (players [(slot, x, y)], ball (x, y) or None)}, normalised, one file pass."""
    wanted = {frame + 1 for frame in frames}  # frame_id counts from 1
    out = {}
    with open(raw_path) as handle:
        for line in handle:
            head = line[:line.index(FIELD_SEP)]
            if int(head) not in wanted:
                continue
            parts = line.split(FIELD_SEP)
            players = []
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if group[0] != MISSING:
                    values = group.split(",")
                    players.append((slot, float(values[0]), float(values[1])))
            ball = None
            if parts[2][0] != MISSING:
                values = parts[2].split(",")
                ball = (float(values[0]), float(values[1]))
            out[int(head) - 1] = (players, ball)
            if len(out) == len(wanted):
                break
    return out


def labels_from(path):
    """Label per frame from a frame,label CSV, or None if the file is not there."""
    if not path or not os.path.isfile(path):
        return None
    return pd.read_csv(path)["label"].astype(str).str.strip().str.upper().values


def draw_pitch(cv2, image, matrix, length, width):
    """Pitch outline, halfway line and centre circle, through the homography."""
    def line(points):
        xs, ys = load_homography.project(matrix, points[:, 0], points[:, 1])
        pts = np.stack([xs, ys], axis=1).round().astype(np.int32)
        cv2.polylines(image, [pts], False, PITCH_COLOUR, 1, cv2.LINE_AA)

    t = np.linspace(0, 1, 50)
    line(np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=float))
    line(np.stack([np.full_like(t, 0.5), t], axis=1))
    angle = np.linspace(0, 2 * np.pi, 72)
    line(np.stack([0.5 + 9.15 * np.cos(angle) / length, 0.5 + 9.15 * np.sin(angle) / width], axis=1))


def draw_ring(cv2, image, matrix, x, y, radius_x, radius_y, colour, thickness):
    """A ring on the pitch plane around (x, y), drawn as the projected ellipse it really is."""
    angle = np.linspace(0, 2 * np.pi, 24)
    xs, ys = load_homography.project(matrix, x + radius_x * np.cos(angle), y + radius_y * np.sin(angle))
    pts = np.stack([xs, ys], axis=1).round().astype(np.int32)
    cv2.polylines(image, [pts], True, colour, thickness, cv2.LINE_AA)


def annotate(cv2, image, frame, matrix, tracking, teams, pitch, gt, submission, detections, boxes,
             names):
    """Everything on one frame. Returns the image."""
    length, width = pitch
    players, ball = tracking.get(frame, ([], None))
    draw_pitch(cv2, image, matrix, length, width)

    nearest = {HOME: (np.inf, None), AWAY: (np.inf, None)}
    if ball is not None:
        for slot, x, y in players:
            distance = np.hypot((x - ball[0]) * length, (y - ball[1]) * width)
            if distance < nearest[teams[slot]][0]:
                nearest[teams[slot]] = (distance, slot)
    closest = min(nearest.values(), key=lambda item: item[0])[1]

    for slot, x, y in players:
        team = teams[slot]
        draw_ring(cv2, image, matrix, x, y, RING_RADIUS_M / length, RING_RADIUS_M / width,
                  TEAM_COLOURS[team], 3 if slot == closest else 1)
        px, py = load_homography.project(matrix, x, y)
        cv2.putText(image, str(slot + 1), (int(px) + 8, int(py) + 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, TEAM_COLOURS[team], 1, cv2.LINE_AA)

    if ball is not None:
        bx, by = load_homography.project(matrix, ball[0], ball[1])
        cv2.circle(image, (int(bx), int(by)), 7, BALL_COLOUR, 2, cv2.LINE_AA)
    for x1, y1, x2, y2, conf in detections.get(frame, []):
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), DETECTION_COLOUR, 1)
        cv2.putText(image, "{:.2f}".format(conf), (int(x1), int(y1) - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, DETECTION_COLOUR, 1, cv2.LINE_AA)
    for cls, conf, x1, y1, x2, y2 in boxes:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), MODEL_COLOUR, 1)
        cv2.putText(image, names[cls], (int(x1), int(y2) + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    MODEL_COLOUR, 1, cv2.LINE_AA)

    def metres(value):
        return "{:.1f} m".format(value) if np.isfinite(value) else "-"

    header = "frame {}   truth {}   predicted {}   ball {}   nearest H {} (#{})   A {} (#{})".format(
        frame, gt[frame] if gt is not None else "-",
        submission[frame] if submission is not None else "-",
        "seen" if ball is not None else "MISSING",
        metres(nearest[HOME][0]), nearest[HOME][1] + 1 if nearest[HOME][1] is not None else "-",
        metres(nearest[AWAY][0]), nearest[AWAY][1] + 1 if nearest[AWAY][1] is not None else "-")
    cv2.rectangle(image, (0, 0), (image.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(image, header, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                cv2.LINE_AA)
    cv2.putText(image, "H blue  A red  ATD ball yellow  detector white  pitch green",
                (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                cv2.LINE_AA)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("frames", help="11351 | 11351,11708 | 11340:11480 | 11340:11480:10")
    parser.add_argument("--data-dir", default=None, help="data bundle, if not ./data or MTAH_DATA_DIR")
    parser.add_argument("--submission", default=os.path.join(ROOT, "evaluation", "results",
                                                             "possession_v3.csv"),
                        help="a frame,label CSV whose label is shown in the header")
    parser.add_argument("--detections", default=None,
                        help="ball detections CSV from export_detections.py, drawn as white boxes")
    parser.add_argument("--weights", default=None,
                        help="run this Ultralytics model on each frame and draw all its boxes")
    parser.add_argument("--mp4", action="store_true", help="write one clip instead of images")
    parser.add_argument("--out", default=None, help="folder to write to (default data/tactical/overlay)")
    args = parser.parse_args()

    try:
        import cv2
    except ImportError:
        sys.exit("needs OpenCV: pip install opencv-python")

    frames = parse_frames(args.frames)
    if not frames:
        sys.exit("no frames in {!r}".format(args.frames))
    meta_path, raw_path, video_path = require(["atd_meta", "atd_raw", "video"], GAME, args.data_dir)
    teams = team_of_each_track(meta_path)
    matrices, _, valid = load_homography.load(GAME, args.data_dir)
    pitch = GAMES[GAME]["pitch"]
    gt = labels_from(GT_PATH.format(GAME))
    submission = labels_from(args.submission)

    detections = {}
    if args.detections:
        table = pd.read_csv(args.detections)
        table = table[table["frame"].isin(frames)]
        for row in table.itertuples(index=False):
            detections.setdefault(int(row.frame), []).append(
                (row.x1, row.y1, row.x2, row.y2, getattr(row, "confidence", 1.0)))

    model = None
    if args.weights:
        from export_detections import detections as model_boxes, load_model
        model = load_model(args.weights)

    print("reading the tracking for {} frame(s) ...".format(len(frames)))
    tracking = read_tracking(raw_path, frames)

    out_dir = args.out or os.path.join(os.path.dirname(video_path), OUT_FOLDER)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    video = cv2.VideoCapture(video_path)
    writer = None
    position = None
    for frame in frames:
        if not valid[frame]:
            print("  frame {}: no homography, skipped".format(frame))
            continue
        if position != frame:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = video.read()
        position = frame + 1
        if not ok:
            print("  frame {}: could not be read from the video, skipped".format(frame))
            continue
        boxes = []
        if model is not None:
            boxes = model_boxes(model(image, imgsz=1280, conf=0.25, verbose=False)[0])
        image = annotate(cv2, image, frame, matrices[frame], tracking, teams, pitch, gt,
                         submission, detections, boxes, model.names if model is not None else {})
        if args.mp4:
            if writer is None:
                path = os.path.join(out_dir, "{}_overlay_{}-{}.mp4".format(GAME, frames[0], frames[-1]))
                writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), MP4_FPS,
                                         (image.shape[1], image.shape[0]))
            writer.write(image)
        else:
            path = os.path.join(out_dir, "{}_overlay_{}.jpg".format(GAME, frame))
            cv2.imwrite(path, image)
            print("  wrote {}".format(path))
    if writer is not None:
        writer.release()
        print("  wrote {}".format(path))


if __name__ == '__main__':
    main()
