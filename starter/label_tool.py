"""Hand-label possession on stretches of the tactical video, to build a validation set.

The public ground truth is 17 possessions -- too few to tell two versions of a system apart.
RULES.md allows a small amount of hand-labelled data to *train or validate* a model, as long as
the submitted labels come out of the algorithm. This tool is for that: you watch a stretch of the
match, mark every change of possession and every stoppage, and get a frame,label file that
evaluation/run_evaluation.py scores against exactly like the official ground truth.

    python starter/label_tool.py --half 1                           # the whole first half
    python starter/label_tool.py --half 2                           # the whole second half
    python starter/label_tool.py --start 30000 --end 37500          # label 5 minutes
    python starter/label_tool.py --start 11185 --end 18684          # practise on the public window
    python starter/label_tool.py --agreement                        # you vs the public truth
    python starter/label_tool.py --export                           # merge every window you did
    python evaluation/run_evaluation.py --gt data/labels/BAR-ATH_validation_gt.csv --bootstrap 1000
    python starter/diagnose.py --keepers-by-side --gt data/labels/BAR-ATH_validation_gt.csv

How labelling works: a label holds from the frame you mark it until the next mark. At the start of
the window, mark what is happening; then mark each moment that changes:

    h   Barcelona (home) has the ball from this frame       space   play / pause
    a   Athletic (away) has the ball from this frame        , .     one frame back / forward
    d   dead ball from this frame                           j l     one second back / forward
    x   delete the mark governing this frame                J L     five seconds back / forward
    m   move the mark governing this frame to this frame    [ ]     one minute back / forward
    n p next / previous mark                                - =     slower / faster playback
    t   tracking overlay on / off                           s       save
    q Esc   save and quit

The whole match: --half 1 / --half 2 label a whole half (its graded frames). A new window starts
with the public ground truth filled in where it overlaps the public window; everything else is
left unlabelled (X, dark on the timeline) for you to label. --prefill FILE fills the rest from a
submission's labels instead (best_so_far.csv; bare names are looked up in evaluation/results),
one mark per change, so you only correct: jump from mark to mark with n,
fix the frame with m, the label with h/a/d, delete a wrong change with x, add a missing one with
h/a/d. The tool opens where you last were (the furthest frame you have reached is saved next to
the marks) and the header shows how far through the window you are. A second strip under the
picture shows +/- 30 s around the cursor, where the marks of a whole half are far enough apart to
see.

Pause and step frame by frame to put a mark on the exact frame: when you mark while the video
plays, your reaction time puts every mark a third of a second late, and spellF1 scores boundaries.

Label the way the ground truth does -- CHALLENGE.md and the public window are the reference, and
practising on the public window first (its ground truth is shown in the header there) is the best
way to learn it. --agreement then scores your public-window labels against the real ground truth
with the official scorer: it says whether you label the way the truth does, and how close to 1.0
a careful human gets -- which is a better target than 1.0 itself. Your own marks inside the public
window are left out of --export, where the real ground truth is used instead (--include-public):
  - D from the moment the ball is out or play is stopped, until the restart is taken
  - possession changes only when the other team gains *control*: a touch, a deflection or a duel
    won without control is not a change, and a loose ball stays with the team that last had it
  - every restart has a clear possessor

Marks are saved to data/labels/ after every change, so nothing is lost if the tool closes. That
folder is gitignored: the labels describe this match's video and are not to be shared or
committed. Everything runs on your machine.

    pip install opencv-python
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import DATA_DIR, GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402

# Params
LABEL_FOLDER = "labels"
MARKS_NAME = "{}_marks_{}-{}.csv"
EXPORT_NAME = "{}_validation_gt.csv"
GT_PATH = os.path.join(ROOT, "evaluation", "gt", "{}_possession_gt_public.csv")
DISPLAY_WIDTH = 1280          # the window is scaled to this width
SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0)
ZOOM_SECONDS = 30             # the lower strip shows this many seconds either side of the cursor
TRACKING_CHUNK = 3000         # frames of tracking read at a time for the overlay (2 min)
PROGRESS_SUFFIX = ".progress"   # next to the marks file: the furthest frame reached
KEYS = {"h": "H", "a": "A", "d": "D"}
# Colours are BGR.
LABEL_COLOURS = {"H": (255, 128, 0), "A": (40, 40, 230), "D": (150, 150, 150), "X": (40, 40, 40)}


def labels_dir(data_dir):
    return os.path.join(data_dir or DATA_DIR, LABEL_FOLDER)


def load_marks(path):
    """{frame: label} from a marks file, {} if there is none yet."""
    if not os.path.isfile(path):
        return {}
    table = pd.read_csv(path)
    return {int(f): str(l).strip().upper() for f, l in zip(table["frame"], table["label"])}


def save_marks(path, marks, end):
    """Marks plus a closing X just after the window, so the window's extent is in the file."""
    folder = os.path.dirname(path)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    rows = sorted((f, l) for f, l in marks.items() if l in ("H", "A", "D", "X") and f <= end)
    rows.append((end + 1, "X"))
    pd.DataFrame(rows, columns=["frame", "label"]).to_csv(path, index=False)


def expand(marks, n_frames):
    """Full-length label array from {frame: label} change points; X before the first mark."""
    labels = np.full(n_frames, "X", dtype="<U1")
    ordered = sorted(marks.items())
    for (frame, label), following in zip(ordered, ordered[1:] + [(n_frames, None)]):
        labels[max(frame, 0):min(following[0], n_frames)] = label
    return labels


def governing(marks, frame):
    """The mark in force at `frame`: the latest one at or before it, or None."""
    before = [f for f in marks if f <= frame]
    return max(before) if before else None


def prefill_marks(name, start, end):
    """{frame: label} at every change inside start..end: the public ground truth inside the
    public window, X (not labelled yet) everywhere else -- or, with name, a submission's labels
    there instead of X."""
    truth = pd.read_csv(GT_PATH.format(GAME))["label"].astype(str).str.strip().str.upper().values
    labels = np.full(len(truth), "X", dtype=object)
    source = "the public ground truth"
    if name:
        path = name if os.path.exists(name) else os.path.join(ROOT, "evaluation", "results",
                                                              os.path.basename(name))
        labels = pd.read_csv(path)["label"].astype(str).str.strip().str.upper().values.copy()
        source += " and " + os.path.relpath(path, ROOT)
    public = truth != "X"
    labels[public] = truth[public]
    labels = labels[start:end + 1]
    marks = {start: labels[0]}
    for i in np.flatnonzero(labels[1:] != labels[:-1]) + 1:
        marks[start + int(i)] = labels[i]
    marks = {f: l for f, l in marks.items() if l in ("H", "A", "D", "X")}
    if all(l == "X" for l in marks.values()):
        return {}, source  # the window does not touch the public one: nothing to fill in
    return marks, source


def with_public_truth(marks, start, end):
    """The marks with the public ground truth laid over the public window -- whatever the marks
    file says there (an older run may have filled it from a submission); outside it nothing
    changes. Returns (marks, whether anything changed)."""
    truth = pd.read_csv(GT_PATH.format(GAME))["label"].astype(str).str.strip().str.upper().values
    public = truth != "X"
    public[:start] = False
    public[end + 1:] = False
    if not public.any():
        return marks, False
    labels = expand(marks, len(truth))
    labels[public] = truth[public]
    window = labels[start:end + 1]
    new = {start: window[0]}
    for i in np.flatnonzero(window[1:] != window[:-1]) + 1:
        new[start + int(i)] = window[i]
    new = {f: l for f, l in new.items() if l in ("H", "A", "D", "X")}
    if all(l == "X" for l in new.values()):
        new = {}
    old = {f: l for f, l in marks.items() if start <= f <= end}
    return new, new != old


def load_progress(path, start):
    try:
        with open(path + PROGRESS_SUFFIX) as handle:
            return int(handle.read().strip())
    except (OSError, ValueError):
        return start


def save_progress(path, frame):
    with open(path + PROGRESS_SUFFIX, "w") as handle:
        handle.write(str(frame))


def export(data_dir, include_public):
    """Merge every marks file into one frame,label ground truth. Later windows win overlaps."""
    n_frames = source_frames("tactical", GAME)
    folder = labels_dir(data_dir)
    paths = sorted(glob.glob(os.path.join(folder, MARKS_NAME.format(GAME, "*", "*"))))
    if not paths and not include_public:
        sys.exit("no marks files in {} yet -- label a window first".format(folder))

    merged = np.full(n_frames, "X", dtype="<U1")
    if include_public:
        public = pd.read_csv(GT_PATH.format(GAME))["label"].astype(str).str.strip().str.upper().values
        merged[public != "X"] = public[public != "X"]
        print("  public window: {} frames".format(int((public != "X").sum())))
    graded = np.zeros(n_frames, dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True
    # The public window has a real ground truth; your marks there are for --agreement only.
    graded &= ~public_mask()

    for path in paths:
        labels = expand(load_marks(path), n_frames)
        labelled = (labels != "X") & graded
        overlap = int((labelled & (merged != "X")).sum())
        merged[labelled] = labels[labelled]
        possessions = count_possessions(labels[labelled])
        print("  {}: {} frames, {} possessions{}".format(
            os.path.basename(path), int(labelled.sum()), possessions,
            "  ({} frames overlap an earlier window and were replaced)".format(overlap)
            if overlap else ""))

    out = os.path.join(folder, EXPORT_NAME.format(GAME))
    if not os.path.isdir(folder):
        os.makedirs(folder)
    pd.DataFrame({"frame": np.arange(n_frames), "label": merged}).to_csv(out, index=False)
    total = int((merged != "X").sum())
    print("\nwrote {}: {} labelled frames ({:.1f} min), {} possessions".format(
        out, total, total / GAMES[GAME]["fps"] / 60.0, count_possessions(merged[merged != "X"])))
    print("score against it:  python evaluation/run_evaluation.py --gt {} --bootstrap 1000".format(out))


def public_mask():
    """True on the frames the public ground truth grades."""
    public = pd.read_csv(GT_PATH.format(GAME))["label"].astype(str).str.strip().str.upper().values
    return public != "X"


def agreement(data_dir):
    """Score your labels of the public window against its ground truth, with the official scorer."""
    import subprocess
    import tempfile
    n_frames = source_frames("tactical", GAME)
    paths = sorted(glob.glob(os.path.join(labels_dir(data_dir), MARKS_NAME.format(GAME, "*", "*"))))
    public = public_mask()
    mine = np.full(n_frames, "X", dtype="<U1")
    for path in paths:
        labels = expand(load_marks(path), n_frames)
        inside = (labels != "X") & public
        mine[inside] = labels[inside]
    covered = int((mine != "X").sum())
    if not covered:
        sys.exit("no labels inside the public window yet: "
                 "python starter/label_tool.py --start 11185 --end 18684")
    print("your labels cover {} of the {} public frames".format(covered, int(public.sum())))
    folder = tempfile.mkdtemp(prefix="label_agreement_")
    pd.DataFrame({"frame": np.arange(n_frames), "label": mine}).to_csv(
        os.path.join(folder, "my_labels.csv"), index=False)
    subprocess.call([sys.executable, os.path.join(ROOT, "evaluation", "run_evaluation.py"),
                     "--results", folder, "--out", folder, "--bootstrap", "1000"])


def count_possessions(labels):
    """Maximal runs of H or of A."""
    if not len(labels):
        return 0
    starts = np.concatenate(([0], np.flatnonzero(labels[1:] != labels[:-1]) + 1))
    return int(np.isin(labels[starts], ("H", "A")).sum())


def strip_of(cv2, labels, marks, lo, hi, frame, width, height):
    """One timeline strip over frames lo..hi: labels as colours, marks as white lines, cursor."""
    strip = np.zeros((height, width, 3), dtype=np.uint8)
    span = max(hi - lo + 1, 1)
    columns = (lo + np.arange(width) * span // width).clip(lo, hi)
    shown = labels[columns]
    for label, colour in LABEL_COLOURS.items():
        strip[2:height - 2, shown == label] = colour
    for mark in marks:
        if lo <= mark <= hi:
            x = int((mark - lo) * width / span)
            cv2.line(strip, (x, 0), (x, height - 1), (255, 255, 255), 1)
    cursor = int((frame - lo) * width / span)
    cv2.line(strip, (cursor, 0), (cursor, height - 1), (0, 255, 255), 2)
    return strip


def draw(cv2, image, frame, start, end, marks, gt, playing, speed, scale, progress=None):
    """Header, current label, the whole window's timeline and a +/- ZOOM_SECONDS one."""
    height, width = image.shape[:2]
    current = marks[governing(marks, frame)] if governing(marks, frame) is not None else "X"
    cv2.rectangle(image, (0, 0), (width, 34), (0, 0, 0), -1)
    fps = GAMES[GAME]["fps"]
    seconds = frame // fps
    header = "frame {}  video {}:{:02d}:{:02d}  label {}  {}  x{}".format(
        frame, seconds // 3600, seconds % 3600 // 60, seconds % 60, current,
        "playing" if playing else "paused", speed)
    if progress is not None:
        header += "  done {:.0f}%".format(100.0 * (progress - start) / max(1, end - start))
    if gt is not None and gt[frame] != "X":
        header += "   public truth {}".format(gt[frame])
    cv2.putText(image, header, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1,
                cv2.LINE_AA)
    cv2.rectangle(image, (width - 60, 4), (width - 8, 30), LABEL_COLOURS[current], -1)

    labels = expand(marks, end + 2)
    whole = strip_of(cv2, labels, marks, start, end, frame, width, 22)
    zoom = ZOOM_SECONDS * fps
    near = strip_of(cv2, labels, marks, max(start, frame - zoom), min(end, frame + zoom), frame,
                    width, 30)
    keys = np.zeros((16, width, 3), dtype=np.uint8)
    cv2.putText(keys, "h/a/d mark  m move mark here  x delete  n/p next/prev mark  space play  "
                ", . j l J L [ ] step  - = speed  t tracking  s save  q quit", (6, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1, cv2.LINE_AA)
    return np.vstack([image, whole, near, keys])


def label(data_dir, start, end, prefill=None, reset=False):
    """The interactive loop."""
    try:
        import cv2
    except ImportError:
        sys.exit("needs OpenCV: pip install opencv-python")
    n_frames = source_frames("tactical", GAME)
    if not 0 <= start <= end < n_frames:
        sys.exit("need 0 <= start <= end < {}".format(n_frames))
    video_path = require(["video"], GAME, data_dir)[0]
    path = os.path.join(labels_dir(data_dir), MARKS_NAME.format(GAME, start, end))
    if reset:  # start the window over: its marks and progress are deleted
        for old in (path, path + PROGRESS_SUFFIX):
            if os.path.isfile(old):
                os.remove(old)
        print("--reset: deleted the window's marks and progress")
    marks = load_marks(path)
    marks.pop(end + 1, None)
    if not marks:  # a new window: the public ground truth where it overlaps, the rest empty
        marks, source = prefill_marks(prefill, start, end)
        if marks:
            save_marks(path, marks, end)
            print("started from {}: {} marks".format(source, len(marks)))
    elif prefill:
        print("the marks file already has marks: --prefill ignored (delete the file to start over)")
    marks, replaced = with_public_truth(marks, start, end)
    if replaced:
        save_marks(path, marks, end)
        print("the public window now shows the public ground truth (it had other labels)")
    print("marks file {} ({} marks so far)".format(path, len(marks)))
    progress = min(max(load_progress(path, start), start), end)
    gt = pd.read_csv(GT_PATH.format(GAME))["label"].astype(str).str.strip().str.upper().values

    tracking = None
    overlay_on = False
    video = cv2.VideoCapture(video_path)
    frame, position, playing, speed_index = progress, None, False, SPEEDS.index(1.0)
    if frame > start:
        print("resuming at frame {} (the furthest you got)".format(frame))
    image = None
    window = "label {} {}-{}".format(GAME, start, end)
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    fps = GAMES[GAME]["fps"]
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
        if overlay_on:
            if tracking is None or frame not in tracking["frames"]:
                lo = max(start, frame - TRACKING_CHUNK // 2)
                tracking = load_tracking(data_dir, lo, min(end, lo + TRACKING_CHUNK))
            shown = with_tracking(cv2, shown, frame, tracking)
        scale = DISPLAY_WIDTH / float(shown.shape[1])
        shown = cv2.resize(shown, (DISPLAY_WIDTH, int(shown.shape[0] * scale)))
        if frame > progress:
            progress = frame
        shown = draw(cv2, shown, frame, start, end, marks, gt, playing, SPEEDS[speed_index], scale,
                     progress)
        cv2.imshow(window, shown)

        delay = max(1, int(1000.0 / (fps * SPEEDS[speed_index]))) if playing else 0
        key = cv2.waitKey(delay) & 0xFF
        if key == 255:  # no key while playing: next frame
            if playing:
                frame = min(frame + 1, end)
                playing = frame < end
            continue
        char = chr(key)
        if char in KEYS:
            marks[frame] = KEYS[char]
            save_marks(path, marks, end)
        elif char == "x":
            mark = governing(marks, frame)
            if mark is not None:
                del marks[mark]
                save_marks(path, marks, end)
        elif char == "m":  # move the governing mark to this frame
            mark = governing(marks, frame)
            if mark is not None and mark != frame:
                marks[frame] = marks.pop(mark)
                save_marks(path, marks, end)
        elif char in "np":
            playing = False
            later = [f for f in marks if f > frame] if char == "n" else [f for f in marks if f < frame]
            if later:
                frame = min(later) if char == "n" else max(later)
        elif char == " ":
            playing = not playing
        elif char in ",.jlJL[]":
            playing = False
            step = {",": -1, ".": 1, "j": -fps, "l": fps, "J": -5 * fps, "L": 5 * fps,
                    "[": -60 * fps, "]": 60 * fps}[char]
            frame = min(max(frame + step, start), end)
        elif char == "-":
            speed_index = max(0, speed_index - 1)
        elif char == "=":
            speed_index = min(len(SPEEDS) - 1, speed_index + 1)
        elif char == "t":
            overlay_on = not overlay_on
        elif char == "s":
            save_marks(path, marks, end)
            save_progress(path, progress)
            print("saved {} marks".format(len(marks)))
        elif char == "q" or key == 27:
            break
    save_marks(path, marks, end)
    save_progress(path, progress)
    cv2.destroyAllWindows()
    labels = expand(marks, end + 1)[start:end + 1]
    print("saved {} marks to {}: {} labelled frames of {}, {} possessions".format(
        len(marks), path, int((labels != "X").sum()), end - start + 1, count_possessions(labels)))


def load_tracking(data_dir, start, end):
    """Tracking for the window, read once, for the overlay."""
    import overlay
    from baseline import team_of_each_track
    import load_homography
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, data_dir)
    print("reading the tracking for frames {}..{} ...".format(start, end))
    matrices, _, _ = load_homography.load(GAME, data_dir)
    return {"frames": overlay.read_tracking(raw_path, range(start, end + 1)),
            "teams": team_of_each_track(meta_path), "matrices": matrices}


def with_tracking(cv2, image, frame, tracking):
    """The overlay.py drawing on one frame, without its header."""
    import overlay
    if tracking is None:
        return image
    annotated = overlay.annotate(cv2, image, frame, tracking["matrices"][frame], tracking["frames"],
                                 tracking["teams"], GAMES[GAME]["pitch"], None, None, {}, [], {})
    return annotated


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", type=int, help="first frame of the window to label")
    parser.add_argument("--end", type=int, help="last frame of the window to label")
    parser.add_argument("--half", type=int, choices=(1, 2), help="label a whole half instead")
    parser.add_argument("--reset", action="store_true",
                        help="delete the window's marks and progress and start it over")
    parser.add_argument("--prefill", default=None, metavar="FILE",
                        help="start an empty window from a submission's labels (e.g. best_so_far.csv)")
    parser.add_argument("--export", action="store_true",
                        help="merge all labelled windows into one frame,label ground truth")
    parser.add_argument("--agreement", action="store_true",
                        help="score your public-window labels against the public ground truth")
    parser.add_argument("--include-public", action="store_true",
                        help="with --export: add the public 5 minutes to your windows")
    parser.add_argument("--data-dir", default=None, help="data bundle, if not ./data or MTAH_DATA_DIR")
    args = parser.parse_args()
    if args.export:
        export(args.data_dir, args.include_public)
    elif args.agreement:
        agreement(args.data_dir)
    elif args.half is not None:
        first, last = [(f, l) for h, f, l in half_video_blocks(GAME) if h == args.half][0]
        label(args.data_dir, first, last, args.prefill, args.reset)
    elif args.start is not None and args.end is not None:
        label(args.data_dir, args.start, args.end, args.prefill, args.reset)
    else:
        parser.error("give --half, or --start and --end, to label a window -- or --export")


if __name__ == '__main__':
    main()
