# storecam - project memory

Safia AI team, ML Engineer hiring round, stage 2: the "Store camera analytics" case study.
Deadline: Thursday 2026-11-12, 23:59 Tashkent time.
Submission: reply to the invitation email with a public GitHub repository link and a short explanation of decisions.

## Intended outcome

One service that reads at least two video streams at once, detects and tracks people, counts IN and OUT at a door line, measures queue and staff zones every second, raises queue and empty-counter alerts, and writes per-minute CSV and JSON-lines alerts.
Everything (sources, lines, zones, thresholds) comes from one config file, and one command runs it.
The README reports accuracy against hand labels and speed honestly, with known failure cases and a scaling note for about 50 cameras.

## Rules from the brief

- Public footage only, with a licence that allows download, processing and showing; never Safia footage, never test against Safia systems.
- No face recognition and no attempt to identify anyone; blur faces in any published frame, GIF or video.
- Videos and weights stay out of git and are fetched by `scripts/download.sh`; versions are pinned.
- A smaller, honest, working result beats a large unfinished one.

## Working rules

- Before each choice, check the alternatives and record them in `docs/DECISIONS.md`.
- Report measurements exactly as measured, including bad ones.
- Do not commit, push, or publish without explicit authorization from Shokhbek.

## Environment

Apple M4 Pro, 24 GB unified memory, macOS 15, no NVIDIA GPU; the detector runs on PyTorch MPS or CPU.
Python 3.12 managed with `uv`.

## State (2026-11-11)

All required parts of the brief are done, measured and pushed.
Bonuses done: batched inference across streams, staff versus customer (D14, E14), the ONNX Runtime export (D15, E15) and the live page (D16).
The last wrong queue alert was removed by moving the queue zone's top edge in front of the chairs and leaving duplicate boxes out of the queue count (D17, E16); `scripts/queue_rules.py` replays saved tracks to try queue rules quickly.
`scripts/evaluate.py` reuses saved runs unless given `--rerun`; always pass `--rerun` after changing code or config.
Load on this machine comes in bursts from other projects' containers; compare speeds with `scripts/benchmark.py --compare` or `scripts/benchmark_docker.sh <seconds> <repeats>`, which alternate the runtimes and record the load per run.
Local-only files (ignored by git): `runs/review/review.md` (29 queue marks waiting for Shokhbek's review, rebuilt by `scripts/review_sheet.py`), `runs/staff_analysis/` (the scratch analysis behind E14) `runs/onnx/` (the ONNX evaluation run behind E15) and `runs/live_review_prep.md` (a study guide for Shokhbek's live review: the path of a frame, where to change common things, expected questions).

## Next steps, in order

Door departures were looked at again on 2026-11-11 and left as they are: the misses are tracking in a crowd at the doorway and people waiting on the mats, 960 px did not help (E9), and with one departure clip any new rule would be tuned on the clip it is measured on (REPORT 5.1).

1. Done 2026-11-11: the speed comparison (E15). On a quiet machine PyTorch was about 1.4 times faster than the ONNX export in the container, so Docker keeps PyTorch (D15 settled); the Mac keeps PyTorch too.
2. Shokhbek reviews `runs/review/review.md`; then update the labels and results, and say in REPORT.md what the review changed.
3. A final read-through of README.md and REPORT.md against `results/`.
4. Draft the submission email for Shokhbek to send; the agent never sends it.
