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
`scripts/evaluate.py` reuses saved runs unless given `--rerun`; always pass `--rerun` after changing code or config.
Local-only files (ignored by git): `runs/review/review.md` (29 queue marks waiting for Shokhbek's review, rebuilt by `scripts/review_sheet.py`), `runs/staff_analysis/` (the scratch analysis behind E14) and `runs/onnx/` (the ONNX evaluation run behind E15).

## Next steps, in order

1. On a quiet machine (check `uptime` and other projects' containers first, never stop them): `scripts/benchmark.py` for the PyTorch weights and with `--weights models/yolo26s_384x640_b2.onnx --out results/benchmark_onnx.json`, plus the detector-only container comparison; update E15, REPORT 5.4 and the README row, and settle the open question in D15 (ONNX by default in Docker).
2. Possibly door departures: only 5 of 13 hand-labelled departures are matched on door_0307_1720.
3. Shokhbek reviews `runs/review/review.md`; then update the labels and results, and say in REPORT.md what the review changed.
4. A final read-through of README.md and REPORT.md against `results/`.
5. Draft the submission email for Shokhbek to send; the agent never sends it.
