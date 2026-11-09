# storecam

Store camera analytics: people counting at a door, queue and counter monitoring, and alerts, over several video streams at once.

Work in progress for the Safia AI team case study (deadline 2026-11-12); the repository is pushed as the work goes so its history shows the process.

Status, 2026-11-09: the service, door counting, zones, alerts, multi-stream reconnect and the evaluation script work; hand labels and the measured results are being completed.
The full report, including the process and who did what, is `REPORT.md`; the options considered for every choice are in `docs/DECISIONS.md`, and every test run is in `docs/EXPERIMENTS.md`.

Built with Claude Code as an AI coding assistant; the brief allows AI tools.

## Run

```sh
uv run storecam run                  # every source in config.yaml at real-time speed
uv run storecam run --offline        # every sampled frame, as fast as possible (evaluation)
uv run storecam run --http 8765      # also serve live counts and open alerts as JSON
uv run storecam run --save-video     # also write annotated video with faces pixelated
uv run pytest                        # unit tests for the counting, zone and alert logic
```

Outputs go to `runs/out/`: `<source>_minutes.csv` per source, `alerts.jsonl`, and `run_summary.json` with speeds.

## Setup

```sh
uv sync
./scripts/download.sh   # videos and weights, about 620 MB, checked by SHA-256
```

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
