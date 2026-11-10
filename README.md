# storecam

Store camera analytics: people counting at a door, queue and counter monitoring, and alerts, over several video streams at once.

Work in progress for the Safia AI team case study (deadline 2026-11-12); the repository is pushed as the work goes so its history shows the process.

Status, 2026-11-11: the service, door counting, zones, alerts, multi-stream reconnect, the Docker setup and the evaluation are working and measured (results below), plus the bonuses: batched inference, staff versus customer, an ONNX Runtime export and a live page; Shokhbek's review of Claude's queue labels is still to come.
The full report, including the process and who did what, is `REPORT.md`; the options considered for every choice are in `docs/DECISIONS.md`, and every test run is in `docs/EXPERIMENTS.md`.

Built with Claude Code as an AI coding assistant; the brief allows AI tools.

## What it looks like

Counter (left: staff zone in green, queue zone in blue) and entrance (red line: door line, the dot marks the inside); 12 s of each at 5 frames per second, from `storecam run --save-video`.
Faces are pixelated by the service itself (top 30% of each person box, D10); a person the detector misses in a frame is not pixelated in that frame.
People the service judges to be staff are drawn in green (D14).

![Counter view with tracked people, zones and live counts](docs/media/counter.gif)
![Entrance view with tracked people and the door line](docs/media/door.gif)

The live page (`--http`), refreshed every second (D16):

![Live page with one card per camera: in, out, queue, staff at the counter and an open queue alert](docs/media/live_page.png)

## Run

```sh
uv run storecam run                  # every source in config.yaml at real-time speed
uv run storecam run --offline        # every sampled frame, as fast as possible (evaluation)
uv run storecam run --http 8765      # also serve a live page at http://127.0.0.1:8765/ (JSON at /status.json)
uv run storecam run --save-video     # also write annotated video with faces pixelated
uv run storecam run --url counter=rtsp://127.0.0.1:8554/counter   # swap a source's address, keep its zones
uv run pytest                        # unit tests for the counting, zone and alert logic
```

Outputs go to `runs/out/`: `<source>_minutes.csv` per source, `alerts.jsonl`, and `run_summary.json` with speeds.

## Reproduce the measurements

```sh
uv run python scripts/evaluate.py --rerun     # runs every clip offline, compares with labels/, writes results/eval/report.json
uv run python scripts/benchmark.py            # speed: 2 streams in real time, then 1, 2 and 4 streams on GPU and CPU
uv run python scripts/calibrate_staff.py counter data/meva/counter_1650.avi   # the staff uniform colour in config.yaml
uv run --group export python scripts/export_onnx.py && uv run python scripts/compare_detectors.py   # ONNX Runtime against PyTorch (E15)
```

Without `--rerun`, `evaluate.py` reuses the run summaries already in `results/eval/` and only recomputes the comparison.
Offline runs are deterministic on one machine, so the report comes out identical (E13).

## Run with Docker

```sh
./scripts/download.sh          # the clips and weights are mounted, not baked into the image
docker compose up --build
```

This starts two RTSP cameras (MediaMTX looping two clips) and the service reading both, writing to `runs/docker/`, with the live page at http://127.0.0.1:8080/ (JSON at /status.json).
Every port is bound to 127.0.0.1.
Docker on macOS cannot use the Apple GPU, so here the detector runs on CPU at about 5 FPS per stream, half the target; the native run above uses the GPU (E12, D13).
Stop the camera container (`docker compose stop cameras`, then `start`) to watch the service reconnect.

## Run with ONNX Runtime (bonus)

```sh
uv run --group export python scripts/export_onnx.py   # writes models/yolo26s_dynamic.onnx and models/yolo26s_384x640_b2.onnx
```

Then set `model.weights` in `config.yaml` to one of them; nothing else changes.
On a Mac the fixed-size export runs on the GPU through CoreML (as FP32), elsewhere on CPU; on all five clips it gave exactly the PyTorch results (E15, D15).

## Setup

```sh
uv sync
./scripts/download.sh   # videos and weights, about 620 MB, checked by SHA-256; the MEVA bucket is slow (about 50 min here), and the script resumes after a dropped connection
```

## Results

Measured on the MEVA clips below against hand labels; full tables, method and failure cases with timestamps are in `REPORT.md` sections 5 and 6.

| What | Result |
|---|---|
| Door, arrival clip, hand vs system IN/OUT | 22/3 vs 21/4 (19 of 25 events matched within 2 s) |
| Door, departure clip, hand vs system IN/OUT | 0/13 vs 0/11 (5 of 13 events matched within 2 s) |
| Queue length MAE at 10 s marks | 1.18 (Shokhbek, 11 marks); 0.77, 0.30, 0.00 (Claude, 30 marks on each of 3 clips) |
| Queue alerts | 3 of 3 expected found (two merged into one), 1 wrong |
| Staff versus customer (bonus) | uniform colour plus time behind the counter; left the attendant out of the queue when they walked through it, with one 1 s mistake after an ID switch (E14) |
| ONNX Runtime export (bonus) | identical results to PyTorch on all five clips; detector 15.6 ms against 17.8 ms per pair of frames on the Mac GPU, and about 3.4 times faster than PyTorch on CPU in Docker, both from single runs on a machine that was not quiet (E15); the full benchmark is still to be repeated |
| Empty-counter alerts | none expected, 0 wrong with the staff rule (1 wrong without it) |
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
