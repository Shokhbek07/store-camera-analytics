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
