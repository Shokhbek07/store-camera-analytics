# Store camera analytics: report

Safia AI team, ML Engineer case study, stage 2.
This report is written as the work goes; sections marked "pending" are filled in when their measurements are done.
Each claim points to its source: decisions are `D1` and onwards in `docs/DECISIONS.md`, experiments are `E1` and onwards in `docs/EXPERIMENTS.md`, and raw result files are in `results/`.

## 1. Summary

Pending: written last, from the measured results.

## 2. The task and how it was approached

The brief asks for a service that turns store-like video into door counts, queue and counter measurements and alerts, over at least two streams at once, with honest measurement against hand labels.
It values "a smaller, honest, working result over a large unfinished one", and work started on 2026-11-08 with four days to the deadline against a task sized at five to six.

So the approach was:

1. Get footage that actually contains the events to measure before writing code, because without real queues and real door traffic nothing can be measured (E1).
2. Before each choice, check the alternatives and measure on our own footage where the published numbers do not fit (D1 to D12).
3. Build the smallest service that meets every required point, with the event logic in a small tested module that can be explained and changed live (D4).
4. Measure against hand labels and report the numbers as they come out, including disagreements between labellers (E7).

## 3. Process timeline

**2026-11-08, footage.**
Pexels and Pixabay clips were too short (10 to 19 s) and mostly handheld.
CAVIAR had a fixed shop-entrance view but no counter.
Sampling all 28 MEVA cameras and counting their activity labels found a school lobby snack counter with real queues and an attendant (G421), and a busy entrance (G420) (E1, D1).

**2026-11-08, model.**
Eight detector and tracker combinations were measured on 30 s of each camera; YOLO26s with ByteTrack broke tracks least for its speed (E2, D2).

**2026-11-08, event logic and service.**
The door line counter with a dead band, per-second zone occupancy and alerts with hysteresis were written with 16 unit tests (D4).
Ultralytics' own multi-stream loader was rejected because one ended stream stops all of them, so each source got its own reader thread feeding one batched detector (D5).

**2026-11-08, geometry.**
Plotting every tracked foot point over the zones showed the first queue zone missed the tail of the queue and revealed the whiteboard occlusion (E3).

**2026-11-08, first runs.**
The whole service ran at 108 FPS for two sources offline (E4); a stream killed mid-run reconnected by itself while the other stream kept running (E5).
The first runs exposed the staff zone reading empty for a few seconds a minute while the attendant was present (E4).

**2026-11-08 to 2026-11-09, labelling.**
A small labeller was written (D9).
Shokhbek hand-labelled both door clips and the first queue marks; Claude labelled the remaining queue marks from frames; the two labellers agree exactly on 4 of 11 shared marks, which shows how much the ground truth itself depends on the counting rule (E7).

## 4. What was built

- `scripts/download.sh`: the five clips and the weights, checked by SHA-256 (D3).
- `src/storecam/sources.py`: one reader thread per source; real-time pacing for files, reconnect with exponential backoff for streams (D5).
- `src/storecam/detect.py`: YOLO26s batched across sources, one ByteTrack per source (D2).
- `src/storecam/events.py`: door line counting, per-second zone sampling, alerts (D4).
- `src/storecam/analytics.py`: per-minute CSV per source and alerts as JSON lines.
- `src/storecam/service.py`: the main loop, a live JSON endpoint, face-pixelated annotated video (D10).
- `config.yaml`: every source, line, zone and threshold.
- `scripts/label.py` and `scripts/evaluate.py`: labelling and comparison with the labels.

## 5. Measurement

Pending: door counts against the hand counts, queue mean absolute error per labeller, expected alerts found, missed and raised wrongly, and speed.

## 6. Known failure cases

Pending, with timestamps. Already seen: the whiteboard hiding lower bodies (E3), the attendant lost at the frame edge (E4), people at the coffee table on the zone edge (E3, E7), and possible stair traffic over the door line (E3).

## 7. Scaling to about 50 cameras

Pending.

## 8. Who did what

Shokhbek chose the direction at each step, labelled both door clips and the first queue marks by hand, and reviews the labels and results.
Claude Code (an AI coding assistant, which the brief allows) wrote the code and documents, ran the experiments, and labels the remaining queue marks and the expected alerts from extracted frames (in progress).
Which labels came from whom is recorded in the label files themselves and in E7.
