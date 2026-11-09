# storecam

Store camera analytics: people counting at a door, queue and counter monitoring, and alerts, over several video streams at once.

Work in progress for the Safia AI team case study (deadline 2026-11-12); the repository is pushed as the work goes so its history shows the process.

Status, 2026-11-10: the service, door counting, zones, alerts, multi-stream reconnect, the Docker setup and the evaluation are working and measured (results below); Shokhbek's review of Claude's queue labels is still to come.
The full report, including the process and who did what, is `REPORT.md`; the options considered for every choice are in `docs/DECISIONS.md`, and every test run is in `docs/EXPERIMENTS.md`.

Built with Claude Code as an AI coding assistant; the brief allows AI tools.

## Run

```sh
uv run storecam run                  # every source in config.yaml at real-time speed
uv run storecam run --offline        # every sampled frame, as fast as possible (evaluation)
uv run storecam run --http 8765      # also serve live counts and open alerts as JSON
uv run storecam run --save-video     # also write annotated video with faces pixelated
uv run storecam run --url counter=rtsp://127.0.0.1:8554/counter   # swap a source's address, keep its zones
uv run pytest                        # unit tests for the counting, zone and alert logic
```

Outputs go to `runs/out/`: `<source>_minutes.csv` per source, `alerts.jsonl`, and `run_summary.json` with speeds.

## Run with Docker

```sh
./scripts/download.sh          # the clips and weights are mounted, not baked into the image
docker compose up --build
```

This starts two RTSP cameras (MediaMTX looping two clips) and the service reading both, writing to `runs/docker/`, with live status at http://127.0.0.1:8080/.
Every port is bound to 127.0.0.1.
Docker on macOS cannot use the Apple GPU, so here the detector runs on CPU at about 5 FPS per stream, half the target; the native run above uses the GPU (E12, D13).
Stop the camera container (`docker compose stop cameras`, then `start`) to watch the service reconnect.

## Setup

```sh
uv sync
./scripts/download.sh   # videos and weights, about 620 MB, checked by SHA-256
```

## Results

Measured on the MEVA clips below against hand labels; full tables, method and failure cases with timestamps are in `REPORT.md` sections 5 and 6.

| What | Result |
|---|---|
| Door, arrival clip, hand vs system IN/OUT | 22/3 vs 21/4 (19 of 25 events matched within 2 s) |
| Door, departure clip, hand vs system IN/OUT | 0/13 vs 0/11 (5 of 13 events matched within 2 s) |
| Queue length MAE at 10 s marks | 1.18 (Shokhbek, 11 marks); 0.77, 0.30, 0.00 (Claude, 30 marks on each of 3 clips) |
| Queue alerts | 3 of 3 expected found (two merged into one), 1 wrong |
| Empty-counter alerts | none expected, 1 wrong |
| Speed, Apple M4 Pro GPU (MPS), YOLO26s 640 px FP32, 1164 MB GPU memory | 2 streams in real time: 9.87 FPS each against a 10 FPS target; as fast as possible: 97 / 119 / 131 FPS total for 1 / 2 / 4 streams; CPU only: 23 / 24 / 28 |

## Footage

All footage comes from the MEVA dataset (Multiview Extended Video with Activities), recorded by Kitware, Inc. for the IARPA DIVA programme.
The people in it are actors following a script, filmed at a training facility.

- Licence: Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/
- Dataset page: https://mevadata.org
- Annotations and metadata: https://gitlab.kitware.com/meva/meva-data-repo

| Local file | Camera | Source URL | Timestamps used |
|---|---|---|---|
| `counter_1650.avi` | school.G421, lobby counter | https://mevadata-public-01.s3.amazonaws.com/drops-123-r13/2018-03-07/16/2018-03-07.16-50-00.16-55-00.school.G421.r13.avi | 00:00 to 05:00 (whole clip) |
| `counter_1720.avi` | school.G421, lobby counter | https://mevadata-public-01.s3.amazonaws.com/drops-123-r13/2018-03-07/17/2018-03-07.17-20-00.17-25-00.school.G421.r13.avi | 00:00 to 05:00 (whole clip) |
| `counter_1735.avi` | school.G421, lobby counter | https://mevadata-public-01.s3.amazonaws.com/drops-123-r13/2018-03-07/17/2018-03-07.17-35-00.17-40-00.school.G421.r13.avi | 00:00 to 05:00 (whole clip) |
| `door_0309_1010.avi` | school.G420, entrance | https://mevadata-public-01.s3.amazonaws.com/drops-123-r13/2018-03-09/10/2018-03-09.10-10-01.10-15-01.school.G420.r13.avi | 00:00 to 05:00 (whole clip) |
| `door_0307_1720.avi` | school.G420, entrance | https://mevadata-public-01.s3.amazonaws.com/drops-123-r13/2018-03-07/17/2018-03-07.17-20-01.17-25-01.school.G420.r13.avi | 00:00 to 05:00 (whole clip) |

Author: Kitware, Inc. and the IARPA DIVA programme (MEVA dataset).
Why these clips and not others: `docs/DECISIONS.md`, D1.
