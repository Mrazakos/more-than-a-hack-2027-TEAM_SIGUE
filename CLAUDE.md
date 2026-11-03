# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Team SIGUE's working copy of the public starter repo for the "More than a Hack 2027" Phase 1 challenge. The task is to label every video frame of one match (BAR-ATH) with who has the ball. The output is one `frame,label` CSV with 199,319 rows and labels `H` (home), `A` (away), `D` (dead ball) and `X` (ungraded or unknown). Entries are ranked on `spellF1`, which counts how many real possessions are recovered as whole possessions. `macroF1` is the per-frame view and the first tie-break. Only the first 5 minutes of ground truth are public (video frames 11185..18684, 17 possessions). The full-match ground truth is private. The rules and metric definitions are in `CHALLENGE.md`, `DATA.md`, `EVALUATION.md`, `SUBMISSION.md` and `RULES.md`.

## Commands

There is no linter. The scorer is the check on accuracy. Two checks guard the refactored labeller (`starter/possession/`) against unintended changes:

```bash
python -m unittest discover tests -v   # fast unit tests, no data needed
python tests/golden.py --check         # 15 flag variants of best_so_far.py must reproduce tests/golden_hashes.json byte for byte (needs data)
python tests/golden.py --record        # only after an INTENDED change to the labels: re-record the hashes and say so in the PR
python tests/golden.py --check --legacy        # same check on the frozen single-file copy
```

```bash
pip install -r requirements.txt            # numpy + pandas: all the scorer needs
pip install -r requirements-starter.txt    # + kloppy 3.18.0, matplotlib, notebook (data loaders/tools)

python evaluation/run_evaluation.py                    # score every *.csv in evaluation/results/ on the public 5 min
python evaluation/run_evaluation.py --bootstrap 1000   # + confidence intervals
python evaluation/run_evaluation.py --gt data/labels/BAR-ATH_validation_gt.csv   # against your own labels (--gt also takes public|private|PATH)
python evaluation/validate_submission.py <file.csv>    # run before every submission; same checks as the official scorer

python starter/baseline.py        # regenerate the four committed baselines (numpy only)
python starter/best_so_far.py     # the team's rule-based system -> evaluation/results/best_so_far.csv
python starter/basic_rules.py     # the same rules built up one at a time (best_so_far's rules are not modified; it only reuses its reader)
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
- `starter/best_so_far.py` is the entry point of an ordered rule-based state machine. It starts from nearest-player-to-ball and layers on rules for goalkeepers, kickoff, touch radius with backdating, ball out of play, corners, goal kicks, goals, free kicks, untagged stoppages, pass-by, dead-ball lead and tail, and a team-shape veto. Its module docstring is the authoritative rule list. Rule numbers are historical, and 3, 10 and 12 were tried and removed, so don't renumber.
- The code is the package `starter/possession/`; `best_so_far.py` only holds the docstring, the CLI hand-off and the legacy `prepare()` / `path_changes` / `load_features` that `basic_rules.py` imports.
  - `config.py`: `Params`, a frozen dataclass with every constant and rule switch. Rules read it; nothing writes it (use `dataclasses.replace`). Never add module-level mutable settings.
  - `tracking.py` (ATD reader + npz cache), `tags.py` (`SmartTags`), `signals.py` (pure numpy helpers), `context.py` (`MatchData` read once, `HalfContext` per half), `report.py`, `trace.py`, `labeller.py` (wires the stages), `cli.py`.
  - `rules/`: `kickoff.py` (1), `touch.py` (2, 11), `restarts.py` (4-6), `stoppages.py` (7, 8, 8b), `pass_loop.py` (the only code that writes labels frame by frame; also 2's backdating and 13), `postprocess.py` (15, 14, 9, then 16 after 17). `smoother.py` is rule 17, the learned H/A/D smoother (coefficients in `smoother_dfl.json`). Rules 16 and 17 are off in `Params()`; `starter/externally_tuned.py --smoother` turns them on. Order matters: event tables first, then the pass, then the postprocess passes.
  - `starter/best_so_far_legacy.py` is the frozen pre-refactor single file, kept for comparison (outputs are named `best_so_far_legacy*.csv`, same feature cache). Don't develop there; `tests/golden.py --check --legacy` proves it still matches.
  - The cache name `best_so_far_features_v4_*.npz` is part of the extraction code in `tracking.py`; bump the version if the extracted features change.
- The tracking is automatic and uncorrected, with gaps and identity switches. The ball is missing on about 41% of graded frames, so any code using ball position has to handle that.

## Things to watch

- This repo is public. `.gitignore` keeps `evaluation/results/*.csv` out (only `baseline_*.csv` are committed), along with full-match ground truth, `data/*`, `*.mp4` and `*.pdf`. A complete submission is effectively a copy of the answer, so don't force-add these.
- `data/` is empty in a fresh clone. The scorer and baselines run on CSVs alone, but the loaders, overlay, review and label tools need the downloaded bundle.
- Tuning uses external data only (organizer clarification): never select settings on the BAR-ATH public window. `tools/` holds the DFL open-data harness: `dfl_data.py` (reader, ground truth), `degrade.py` (hides the ball the way the ATD does, by tempo and pitch zone), `tune_dfl.py` (leave-one-match-out parameter search, `--workers`, `--truth both`), `fit_smoother.py` (fits rule 17). Their results go to `experiments/external_log.jsonl`, never `experiments/log.jsonl`. The DFL files live in the gitignored `data_external/`.
- `RULES.md` allows only a small amount of hand-labelled data to train or validate. The submitted labels must come out of the algorithm. Read it before adding learned components or hand-tuned labels.
- The public window has only 17 possessions, so differences between versions are often noise. Compare on `spellF1` and transition F1, not `macroF1` alone, and use `--bootstrap` or a larger `label_tool.py` validation set.
