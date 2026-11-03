# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Team SIGUE's working copy of the public starter repo for the "More than a Hack 2027" Phase 1 challenge. The task is to label every video frame of one match (BAR-ATH) with who has the ball. The output is one `frame,label` CSV with 199,319 rows and labels `H` (home), `A` (away), `D` (dead ball) and `X` (ungraded or unknown). Entries are ranked on `spellF1`, which counts how many real possessions are recovered as whole possessions. `macroF1` is the per-frame view and the first tie-break. Only the first 5 minutes of ground truth are public (video frames 11185..18684, 17 possessions). The full-match ground truth is private. The rules and metric definitions are in `CHALLENGE.md`, `DATA.md`, `EVALUATION.md`, `SUBMISSION.md` and `RULES.md`.

## Commands

There is no test suite or linter. The scorer is the check.

```bash
pip install -r requirements.txt            # numpy + pandas: all the scorer needs
pip install -r requirements-starter.txt    # + kloppy 3.18.0, matplotlib, notebook (data loaders/tools)

python evaluation/run_evaluation.py                    # score every *.csv in evaluation/results/ on the public 5 min
python evaluation/run_evaluation.py --bootstrap 1000   # + confidence intervals
python evaluation/run_evaluation.py --gt data/labels/BAR-ATH_validation_gt.csv   # against your own labels (--gt also takes public|private|PATH)
python evaluation/validate_submission.py <file.csv>    # run before every submission; same checks as the official scorer

python starter/baseline.py        # regenerate the four committed baselines (numpy only)
python starter/best_so_far.py     # the team's rule-based system -> evaluation/results/best_so_far.csv
python starter/basic_rules.py     # the same rules built up one at a time (best_so_far.py stays untouched)
```

`best_so_far.py` has an `--no-<rule>` or numeric flag for each rule (`--no-backdate`, `--no-goal-kick`, `--dead-lead N`, `--shape-veto M` and so on; see `--help`). `--trace FROM TO` prints its decisions frame by frame.

Debugging tools in `starter/` (all need the gated data):
- `review.py` plays the video next to your labels and the truth, with n/p to jump between mistakes.
- `overlay.py FRAME[,FRAME|START:END:STEP]` draws the tracking on the video frames.
- `error_breakdown.py --gt ... --submission ...` splits errors into wrong team, missed dead and false dead.
- `label_tool.py` is for hand-labelling validation windows. `--export` merges them into `data/labels/BAR-ATH_validation_gt.csv`.
- `diagnose.py` currently fails on import (`ModuleNotFoundError: possession_v1`) because it depends on state-machine modules that are not in the repo. Don't rely on it without fixing that.

## Architecture

- `games.py` is the single source of per-match constants: frame count, fps, pitch size, the per-half ATD offsets, half spans, kickoff anchors and asset paths. It is stdlib only. Data is read from `./data` or `$MTAH_DATA_DIR`.
- There are three timelines, and mixing them up is the main source of bugs:
  - A submission and the homography use video frames `0..199318`.
  - The tracking data (ATD) uses `frame_id` `1..199319`. Convert with `id_frame = video_frame + OFFSETS[half]`; the two halves differ by one frame, so a single offset is not exact.
  - Smart Tagging uses seconds from the first video frame: `video_frame = round(seconds * 25)`.
  - The broadcast video is not aligned to the tactical video.
- `starter/load_atd.py`, `load_smart_tagging.py` and `load_homography.py` put the three data sources onto video frames. `utils.py` has the pitch plots.
- `evaluation/possession_metrics.py` defines every metric. `run_evaluation.py` is the official scorer, and the same file produces the official result.
- `starter/best_so_far.py` is an ordered rule-based state machine. It starts from nearest-player-to-ball and layers on rules for goalkeepers, kickoff, touch radius with backdating, ball out of play, corners, goal kicks, goals, free kicks, untagged stoppages, pass-by, dead-ball lead and tail, and a team-shape veto. Its module docstring is the authoritative rule list. Rule numbers are historical, and 3, 10 and 12 were tried and removed, so don't renumber.
- The tracking is automatic and uncorrected, with gaps and identity switches. The ball is missing on about 41% of graded frames, so any code using ball position has to handle that.

## Things to watch

- This repo is public. `.gitignore` keeps `evaluation/results/*.csv` out (only `baseline_*.csv` are committed), along with full-match ground truth, `data/*`, `*.mp4` and `*.pdf`. A complete submission is effectively a copy of the answer, so don't force-add these.
- `data/` is empty in a fresh clone. The scorer and baselines run on CSVs alone, but the loaders, overlay, review and label tools need the downloaded bundle.
- `RULES.md` allows only a small amount of hand-labelled data to train or validate. The submitted labels must come out of the algorithm. Read it before adding learned components or hand-tuned labels.
- The public window has only 17 possessions, so differences between versions are often noise. Compare on `spellF1` and transition F1, not `macroF1` alone, and use `--bootstrap` or a larger `label_tool.py` validation set.
