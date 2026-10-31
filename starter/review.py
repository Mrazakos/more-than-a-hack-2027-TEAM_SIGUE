"""Watch the match next to your labels and the truth, and jump from one mistake to the next.

Opens two windows -- drag each onto its own monitor:

  video   the tactical video, with the tracking drawn on it (players in their team's colour, the
          ball) when you press t
  data    "truth" (the public ground truth) and "predicted" (the file you review) on a +/- 30 s
          timeline, mistakes in red; the whole
          window as one strip; and the numbers behind the frame: is the ball seen, how fast,
          how far is each team's nearest player, how fast are the players moving

    python starter/review.py                       # best_so_far.csv (make it: python starter/best_so_far.py)
    python starter/review.py --submission evaluation/results/baseline_plus_k_h_r2_o_c10_sp_ms_ko.csv
    python starter/review.py --start 15800 --end 16500

    n p     next / previous mistake          space   play / pause
    , .     one frame back / forward         - =     slower / faster
    j l     one second back / forward        t       tracking on the video on / off
    J L     five seconds back / forward      q Esc   quit

The mistakes are also printed in the console when it starts, so you can note them down.

Only the public window has a ground truth; elsewhere only your labels are shown. Everything runs
on your machine, and nothing it shows may leave it: describe what you see, do not share images.

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
import best_so_far  # noqa: E402  -- its tracking reader, cache and tempo

# Params
GT_PATH = os.path.join(ROOT, "evaluation", "gt", "{}_possession_gt_public.csv")
# The official baseline ships in the repository, so the review works before you have made anything.
DEFAULT_SUBMISSION = os.path.join(ROOT, "evaluation", "results", "best_so_far.csv")
PUBLIC = (11185, 18684)       # the public window's graded frames
MIN_MISTAKE = 10              # frames; shorter disagreements are not listed
SPAN = 750                    # frames either side of the cursor on the timeline (30 s)
PANEL = (1280, 560)           # data window, pixels
VIDEO_WIDTH = 1280            # video window width, pixels
SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0)
# BGR
COLOURS = {"H": (255, 128, 0), "A": (40, 40, 230), "D": (150, 150, 150), "X": (45, 45, 45)}
WRONG = (0, 0, 255)
TEXT = (235, 235, 235)


def read_labels(path):
    return pd.read_csv(path)["label"].astype(str).str.strip().str.upper().values


def mistakes(truth, labels, start, end):
    """[(first, last, truth, label)] for every disagreement of MIN_MISTAKE+ frames in start..end."""
    wrong = (truth != labels) & (truth != "X")
    wrong[:start] = False
    wrong[end + 1:] = False
    edges = np.diff(np.concatenate(([0], wrong.astype(np.int8), [0])))
    out = []
    for first, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if stop - first >= MIN_MISTAKE:
            out.append((int(first), int(stop - 1), truth[first], labels[first]))
    return out


def kind(truth, label):
    if truth in "HA" and label == "D":
        return "predicted dead, play is on"
    if truth == "D" and label in "HA":
        return "predicted play, ball is dead"
    if truth in "HA" and label in "HA":
        return "wrong team"
    return "abstained"


def strip(cv2, image, y, height, labels, first, last, colour_of):
    """One horizontal band, frames first..last across the panel width."""
    width = image.shape[1] - 20
    columns = first + (np.arange(width) * (last - first + 1) // width)
    columns = columns.clip(0, len(labels) - 1)
    for label, colour in colour_of.items():
        mask = labels[columns] == label
        image[y:y + height, 10:10 + width][:, mask] = colour


def panel(cv2, frame, start, end, truth, labels, values, errors, current, playing, speed,
          name=""):
    """The data window for one frame."""
    width, height = PANEL
    image = np.full((height, width, 3), 20, dtype=np.uint8)
    fps = GAMES[GAME]["fps"]

    def text(s, x, y, scale=0.6, colour=TEXT):
        cv2.putText(image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)

    t, p = truth[frame], labels[frame]
    text("frame {}   {:+.1f} s   {}   x{}".format(frame, (frame - start) / float(fps),
                                                 "playing" if playing else "paused", speed),
         10, 28, 0.7)
    text("truth", 10, 66); cv2.rectangle(image, (80, 48), (130, 74), COLOURS[t], -1)
    text(t, 98, 67, 0.7, (0, 0, 0))
    text("predicted", 150, 66); cv2.rectangle(image, (250, 48), (300, 74), COLOURS[p], -1)
    text(p, 268, 67, 0.7, (0, 0, 0))
    if t != "X":
        text("OK" if t == p else "WRONG: " + kind(t, p), 330, 67, 0.7,
             (80, 220, 80) if t == p else WRONG)

    # +/- SPAN timeline: truth, yours, mistakes.
    lo, hi = frame - SPAN, frame + SPAN
    text("truth (public ground truth)", 10, 108, 0.5)
    strip(cv2, image, 114, 26, truth, lo, hi, COLOURS)
    text("predicted ({})".format(name), 10, 158, 0.5)
    strip(cv2, image, 164, 26, labels, lo, hi, COLOURS)
    wrong = np.where((truth != labels) & (truth != "X"), "W", "-")
    strip(cv2, image, 194, 8, wrong, lo, hi, {"W": WRONG})
    middle = width // 2
    cv2.line(image, (middle, 110), (middle, 206), (0, 255, 255), 2)
    for second in range(-30, 31, 5):
        x = 10 + int((second * fps + SPAN) * (width - 20) / (2 * SPAN + 1))
        cv2.line(image, (x, 206), (x, 212), TEXT, 1)
        text("{:+d}s".format(second), min(max(x - 14, 0), width - 36), 226, 0.4)

    # Whole window.
    text("whole window, mistakes in red", 10, 256, 0.5)
    strip(cv2, image, 262, 14, labels, start, end, COLOURS)
    strip(cv2, image, 278, 6, np.where((truth != labels) & (truth != "X"), "W", "-"),
          start, end, {"W": WRONG})
    x = 10 + int((frame - start) * (width - 20) / max(1, end - start))
    cv2.line(image, (x, 258), (x, 288), (0, 255, 255), 2)

    # The numbers behind this frame.
    def metres(v):
        return "{:.1f} m".format(v) if np.isfinite(v) else "-"
    rows = [("ball", "seen" if values["seen"] else "MISSING"),
            ("ball speed", "{:.1f} m/s".format(values["speed"]) if np.isfinite(values["speed"]) else "-"),
            ("nearest Barcelona (H)", metres(values["home"])),
            ("nearest Athletic (A)", metres(values["away"])),
            ("players' tempo", "{:.2f} m/s (walking below 1.2)".format(values["tempo"])
             if np.isfinite(values["tempo"]) else "-")]
    for i, (name, value) in enumerate(rows):
        text(name, 10, 322 + 26 * i, 0.55)
        text(value, 260, 322 + 26 * i, 0.55)

    # The mistake list around the current one.
    text("mistakes ({} of {}+ frames)".format(len(errors), MIN_MISTAKE), 640, 322, 0.55)
    for row, k in enumerate(range(max(0, current - 2), min(len(errors), current + 5))):
        first, last, tt, pp = errors[k]
        line = "{:>2}. {}..{}  {:.1f}s  truth {} / predicted {}".format(
            k + 1, first, last, (last - first + 1) / float(fps), tt, pp)
        text(line, 640, 348 + 24 * row, 0.5, (0, 255, 255) if k == current else TEXT)

    text("n p mistake   space play   , . frame   j l 1 s   J L 5 s   - = speed   t tracking   q quit",
         10, height - 14, 0.45, (170, 170, 170))
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--submission", default=DEFAULT_SUBMISSION,
                        help="the frame,label file to review (default: best_so_far.csv)")
    parser.add_argument("--gt", default=None, help="ground truth (default: the public window)")
    parser.add_argument("--start", type=int, default=PUBLIC[0])
    parser.add_argument("--end", type=int, default=PUBLIC[1])
    parser.add_argument("--data-dir", default=None, help="data bundle, if not ./data or MTAH_DATA_DIR")
    args = parser.parse_args()

    try:
        import cv2
    except ImportError:
        sys.exit("needs OpenCV: pip install opencv-python")
    if not os.path.isfile(args.submission):  # a bare name: look in evaluation/results
        args.submission = os.path.join(ROOT, "evaluation", "results", os.path.basename(args.submission))
    if not os.path.isfile(args.submission):
        sys.exit("{} not found -- make it first: python starter/best_so_far.py".format(
            args.submission))

    labels = read_labels(args.submission)
    truth = read_labels(args.gt or GT_PATH.format(GAME))
    start, end = args.start, args.end
    errors = mistakes(truth, labels, start, end)
    fps = GAMES[GAME]["fps"]
    print("{}: {} mistakes of {}+ frames in {}..{}".format(
        os.path.basename(args.submission), len(errors), MIN_MISTAKE, start, end))
    for k, (first, last, t, p) in enumerate(errors):
        print("  {:>2}. {:>6}..{:<6} {:>5.1f} s   truth {} / predicted {}   {}".format(
            k + 1, first, last, (last - first + 1) / float(fps), t, p, kind(t, p)))

    features = best_so_far.load_features(args.data_dir)
    name = os.path.basename(args.submission)
    # keeper slots by side (rule 0) unless the file was made without it
    keepers = "_k" in name if name.startswith("baseline") else "nokeepers" not in name
    home = features["home_keeper" if keepers else "home_fixed"]
    away = features["away_keeper" if keepers else "away_fixed"]
    speed = best_so_far.ball_kinematics(features["ball_x"], features["ball_y"], fps)[0]
    tempo = best_so_far.tempo(features)

    tracking = None
    show_tracking = False
    video = cv2.VideoCapture(require(["video"], GAME, args.data_dir)[0])
    cv2.namedWindow("video", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("data", cv2.WINDOW_AUTOSIZE)
    frame, position, playing, speed_index, current = start, None, False, SPEEDS.index(1.0), 0
    image = None
    while True:
        if position != frame:
            if position is None or frame != position + 1:
                video.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = video.read()
            if not ok:
                print("could not read frame {}".format(frame))
                break
            position = frame
        shown = image.copy()
        if show_tracking:
            if tracking is None:
                import label_tool
                tracking = label_tool.load_tracking(args.data_dir, start, end)
            import label_tool
            shown = label_tool.with_tracking(cv2, shown, frame, tracking)
        scale = VIDEO_WIDTH / float(shown.shape[1])
        cv2.imshow("video", cv2.resize(shown, (VIDEO_WIDTH, int(shown.shape[0] * scale))))
        values = {"seen": not np.isnan(features["ball_x"][frame]), "speed": speed[frame],
                  "home": home[frame], "away": away[frame], "tempo": tempo[frame]}
        cv2.imshow("data", panel(cv2, frame, start, end, truth, labels, values, errors, current,
                                 playing, SPEEDS[speed_index],
                                 os.path.splitext(os.path.basename(args.submission))[0]))

        delay = max(1, int(1000.0 / (fps * SPEEDS[speed_index]))) if playing else 0
        key = cv2.waitKey(delay) & 0xFF
        if key == 255:
            if playing:
                frame = min(frame + 1, end)
                playing = frame < end
            continue
        char = chr(key)
        if char in "np" and errors:
            current = (current + (1 if char == "n" else -1)) % len(errors)
            frame, playing = max(start, errors[current][0] - fps), False
        elif char == " ":
            playing = not playing
        elif char in ",.jlJL":
            playing = False
            step = {",": -1, ".": 1, "j": -fps, "l": fps, "J": -5 * fps, "L": 5 * fps}[char]
            frame = min(max(frame + step, start), end)
        elif char == "-":
            speed_index = max(0, speed_index - 1)
        elif char == "=":
            speed_index = min(len(SPEEDS) - 1, speed_index + 1)
        elif char == "t":
            show_tracking = not show_tracking
        elif char == "q" or key == 27:
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
