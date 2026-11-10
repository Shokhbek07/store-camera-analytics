# Store camera analytics: report

Safia AI team, ML Engineer case study, stage 2.
Each claim points to its source: decisions are `D1` and onwards in `docs/DECISIONS.md`, experiments are `E1` and onwards in `docs/EXPERIMENTS.md`, and raw result files are in `results/`.

## 1. Summary

The service reads two video sources at once, detects people with YOLO26s, tracks them with ByteTrack, and writes per-minute CSV rows and alert events, all from one config file and one command.
On five licensed MEVA clips (D1), measured against hand labels:

- Door: 21 in and 4 out against 22 and 3 hand-counted on the arrival clip, and 0 in and 11 out against 0 and 13 on the departure clip.
  On two clips not used for tuning the count error fell from 4 to 1 and from 6 to 3 (E10).
- Queue length: mean absolute error 1.18 people against Shokhbek's 11 marks, and 0.77, 0.30 and 0.00 against Claude's 30 marks on each of the three counter clips.
  The two labellers themselves differ by 0.82 on average on the marks they share, so the reference is not much sharper than the system.
- Alerts: every expected queue alert was raised, but two of them were merged into one, and one alert was wrong (people at a coffee table on the zone edge).
  A second wrong alert, the attendant counted as a waiting customer, was removed by telling staff from customers by uniform colour and time behind the counter (bonus, D14), narrowly and with limits explained in E14.
- Reproducible: a clean clone, the download script and `scripts/evaluate.py --rerun` give the same numbers (E13).
- Speed on an Apple M4 Pro GPU: two streams in real time held 9.87 of the 10 FPS target each; processed as fast as possible, 2 streams reach 119 FPS and 4 reach 131 FPS in total (E11b).
- An ONNX Runtime export of the detector (bonus) gave exactly the PyTorch results on all five clips and ran the detector about 12% faster on the Mac GPU through CoreML; on CPU in the Docker container it was about 3.4 times faster than PyTorch, measured under load (E15, D15).
- A live page shows each camera's counts and open alerts, refreshed every second (bonus, D16).

The weakest parts are the departure side of the door (5 of 13 events matched in time) and telling staff from customers.

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

**2026-11-10, door fixes and speed.**
Listing the door events side by side showed why departures were missed; a doorway region fixed most of it, checked on two clips not used for tuning (E9, E10).
The speed benchmark was rerun once another project's containers had stopped loading the machine (E11a, E11b).

**2026-11-10, Docker.**
The compose setup first built an 18 GB image and showed RTSP packet loss over UDP; both were fixed, and in the container the detector reaches about half the target frame rate on CPU (E12).

**2026-11-10, clean clone and staff versus customer.**
Following the README from a clean clone found a download that could not resume and a reproducibility check that had reused cached runs; both were fixed and the numbers reproduced from scratch (E13).
Uniform colour alone mostly removed customers in light coats from the queue; adding "has spent time behind the counter" kept the one real case and removed the wrong exclusions (E14, D14).

**2026-11-11, ONNX Runtime and the live page.**
The CoreML set-up that Ultralytics chooses computed in FP16, changed a few boxes and was slower than PyTorch; switching it to an FP32 MLProgram made the ONNX export both exact and faster (E15, D15).
On the Mac CPU ONNX Runtime was slower than PyTorch, but in the Linux container it was about 3.4 times faster, which matters because the container has no GPU.
The live page went on the existing endpoint as one self-contained page (D16).

## 4. What was built

- `scripts/download.sh`: the five clips and the weights, checked by SHA-256 (D3).
- `src/storecam/sources.py`: one reader thread per source; real-time pacing for files, reconnect with exponential backoff for streams (D5).
- `src/storecam/detect.py`: YOLO26s batched across sources, one ByteTrack per source (D2); PyTorch weights or an ONNX export run by ONNX Runtime (D15).
- `scripts/export_onnx.py`: the ONNX exports (D15, E15).
- `src/storecam/events.py`: door line counting, per-second zone sampling, alerts (D4).
- `src/storecam/staff.py` and `scripts/calibrate_staff.py`: staff versus customer by uniform colour and time in the staff zone (D14).
- `src/storecam/analytics.py`: per-minute CSV per source and alerts as JSON lines.
- `src/storecam/service.py`: the main loop, the live page and JSON endpoint (`src/storecam/web.py`, D16), face-pixelated annotated video (D10).
- `config.yaml`: every source, line, zone and threshold.
- `scripts/label.py` and `scripts/evaluate.py`: labelling and comparison with the labels.
- `Dockerfile`, `docker-compose.yml`, `docker/mediamtx.yml`: one-command run with two looping RTSP cameras, detector on CPU (D13, E12).
- `scripts/benchmark.py` and `scripts/review_sheet.py`: speed measurements and a sheet for checking labels.

## 5. Measurement

All numbers come from `scripts/evaluate.py`, which runs the service offline on each clip (every sampled frame is processed, so results are repeatable) and writes `results/eval/report.json`.
Who labelled what is recorded in each file under `labels/` and in E7.

### 5.1 Door counts

Hand counts by Shokhbek, from a key press each time someone crossed the inner edge of the entrance mats (D7, D9).
A system event is "matched" when a hand event in the same direction lies within 2 s.

| Clip | Hand IN/OUT | System IN/OUT | Matched within 2 s | Extra | Missed |
|---|---|---|---|---|---|
| `door_0309_1010` (arrivals) | 22 / 3 | 21 / 4 | 19 | 6 | 6 |
| `door_0307_1720` (departures) | 0 / 13 | 0 / 11 | 5 | 6 | 8 |

The totals are close, but on the departure clip they are close partly by luck: only 5 of the 13 departures are matched within 2 s, and widening the window to 5 or 10 s still matches only 8 (the same matching as `evaluate.py` with the window changed).
Three system counts come 30 to 40 s early (0:03, 0:04 and 0:13), from people who stepped onto the mats and waited there before leaving (failure case 6).
The group that left between 0:42 and 0:54 was counted at 0:39 to 0:46 and 0:56, fewer and earlier than the hand count, as the tracker lost people in the crowd at the doorway; a 960 px input did not change this (E9).
System crossings also come 1 to 2.5 s before the key presses, the delay of a person pressing a key while watching; at 3 s the arrivals match 22 of 25 (E9).

The settings were chosen on these two clips (E9), so the same settings were checked on two door clips not used for tuning, against MEVA's own annotations (E10):

| Clip | MEVA IN/OUT | System IN/OUT |
|---|---|---|
| 2018-03-13 16:30 | 40 / 3 | 41 / 3 |
| 2018-03-12 10:00 | 4 / 4 (UMD-IBM), 7 / 2 (CMU) | 7 / 4 |

### 5.2 Queue length every 10 s

The queue zone count is the median of the frames in each second (D4); the labels count customers waiting at or for the counter at each 10 s mark.

| Clip | Labeller | Marks | MAE | Mean error (system minus hand) | Exactly right | Mean queue |
|---|---|---|---|---|---|---|
| `counter_1650` | Shokhbek | 11 (0:10 to 3:30) | 1.18 | +1.18 | 3 | 1.7 |
| `counter_1650` | Claude | 30 (0:00 to 4:50) | 0.77 | +0.50 | 12 | 2.4 |
| `counter_1720` | Claude | 30 | 0.30 | +0.23 | 23 | 3.6 |
| `counter_1735` | Claude | 30 | 0.00 | 0.00 | 30 | 0.5 |

The system never counts fewer people than Shokhbek; the extra people are mostly on the coffee-table edge of the zone and people walking through it.
On the 11 marks both labellers share, they agree exactly on 4 and Claude counts 0.82 more on average; the difference is the counting rule: Claude counted everyone standing inside the zone outline, Shokhbek counted the people who were queuing (E7).
So "queue length" depends on the counting rule almost as much as on the detector, and the 1.18 against Shokhbek is the more conservative figure.
`counter_1735` is quiet (half a person on average), so its 0.00 shows that empty and near-empty scenes are handled, not that the hard case is solved.

Whether staff was present agreed with the labels at 10 of 11, 27 of 30, 29 of 30 and 26 of 30 marks.

### 5.3 Alerts

Expected alerts were listed by Claude from the 10 s labels and frames (queue: at least 3 waiting for 10 s; empty counter: no staff for 10 s while someone waits).
A system alert is "found" when it overlaps an expected one.

| Clip | Kind | Expected | Raised | Found | Missed | Wrong |
|---|---|---|---|---|---|---|
| `counter_1650` | queue | 2 | 2 | 2 | 0 | 1 |
| `counter_1650` | empty counter | 0 | 0 | 0 | 0 | 0 |
| `counter_1720` | queue | 1 | 1 | 1 | 0 | 0 |
| `counter_1720` | empty counter | 0 | 0 | 0 | 0 | 0 |
| `counter_1735` | queue | 0 | 0 | 0 | 0 | 0 |
| `counter_1735` | empty counter | 0 | 0 | 0 | 0 | 0 |

- `counter_1650`: the expected queue alerts at 0:00 to 0:30 and 0:40 to 1:50 were both covered by one system alert from 0:00 to 1:19 (peak 6), which bridged the short dip and ended about 30 s early.
  The second system alert, 4:14 to 4:47 (peak 4), is wrong: the labels show 1 or 2 people waiting.
- `counter_1720`: one expected alert, 0:20 to 4:20; the system raised it from 0:09 to 4:10 (peak 7).
- `counter_1735`: the attendant really did leave the counter from about 2:35 to 2:45, but nobody was waiting.
  Without the staff rule (D14) the system counted the attendant, crossing the queue zone, and a passer-by as waiting customers and raised a wrong empty-counter alert at 2:44.
  With it, the attendant is left out of the queue for the first two seconds, which keeps the condition under its 10 s threshold; the attendant's track then broke and the new track counted as a customer again, so this fix is narrow (E14).

There are no examples of a real empty-counter alert in this footage, so that alert is only tested by the unit tests and by this one near-miss.

### 5.4 Speed

Machine: Apple M4 Pro (14 CPU cores, 24 GB unified memory), macOS 15, PyTorch on the MPS GPU backend.
Model: YOLO26s, 10.0 million parameters, 640 px input, FP32 (the service uses FP16 only on CUDA; FP16 on MPS was not measured).
Source video: 1920x1072 at 30 fps; 10 frames per second per source are analysed.

| Run | Total FPS | Per stream | Detector ms per batch | GPU memory |
|---|---|---|---|---|
| 2 streams in real time, 120 s (target 10 per stream) | 19.8 | 9.87 each, 1.3% of sampled frames dropped | 16.7 | 1164 MB |
| 1 stream, as fast as possible, GPU (MPS) | 96.8 | 96.8 | 9.8 | 1181 MB |
| 2 streams, as fast as possible, GPU | 119.1 | 59.6 each | 16.1 | 1181 MB |
| 4 streams, as fast as possible, GPU | 131.0 | 32.8 each | 29.1 | 1197 MB |
| 1 stream, CPU only | 23.1 | 23.1 | 42.7 | |
| 2 streams, CPU only | 23.7 | 11.9 each | 83.3 | |
| 4 streams, CPU only | 27.8 | 7.0 each | 141.7 | |

Process memory was about 790 MB for one stream (E2).
The real-time pair uses about a sixth of what the GPU can do, and batching makes each extra stream cheaper (9.8 ms per image alone, 7.3 ms in a batch of 4), so this laptop could analyse about 13 streams at 10 FPS if decoding keeps up.
On CPU alone two streams at 10 FPS just fit.
These are from a quiet machine (E11b, `results/benchmark.json`).
A first run under another project's CPU load reached about half to three quarters of these figures on the GPU and a third on CPU (E11a), which is worth knowing for any shared server.

**ONNX Runtime export (bonus, E15).**
On 40 pairs of real frames, the detector took 15.6 ms per pair as an ONNX export on the Mac GPU (ONNX Runtime with CoreML, FP32) against 17.8 ms in PyTorch, and found the same boxes; the full evaluation gave exactly the PyTorch results on all five clips.
On CPU in the Docker container it took 497 ms per pair against 1675 ms for PyTorch, measured while another project loaded the machine heavily, so only the ratio of about 3.4 is meaningful.
These were detector-only timings on a machine that was not quiet; the full benchmark above is still to be repeated with the ONNX export.

## 6. Known failure cases

Times are positions in the clip (minutes:seconds).

1. **Attendant lost at the left edge of the counter view.**
   The attendant often stands at the frame edge or bends below the counter, so the staff zone reads empty: `counter_1650` at 0:09 to 0:14, 1:06 to 1:09, 3:50 to 3:51 and 4:28 to 4:35, and `counter_1720` at 1:04 to 1:05 and 2:29 to 2:31.
   The 10 s alert delay absorbs these, but the per-minute "empty counter seconds" column overstates the time the counter was unattended.
2. **Staff counted as customers.**
   When the attendant leaves the counter they can walk through the queue zone and be counted as a waiting customer (`counter_1735`, 2:35 to 2:45, which raised a wrong empty-counter alert before D14).
   The staff rule (uniform colour plus time behind the counter) catches this only while the attendant keeps the track ID they had at the counter: here the ID changed after two seconds, and at 4:34 a switch the other way left a customer out of the queue for one second (E14).
3. **People at the coffee table on the zone edge.**
   People standing at the coffee table to the right of the counter are inside or on the edge of the queue zone; this is the main source of the over-count in 5.2 and of the wrong queue alert on `counter_1650` at 4:14 to 4:47.
4. **Whiteboard hiding lower bodies.**
   In the counter view a whiteboard in the foreground cuts boxes off at its top edge, so foot points sit too high and people behind it can fall outside the zone (E3).
5. **Flip-flops at the door line.**
   When the door or another person hides part of a box, the foot point jumps across the line: for example track 216 at 4:45.7 and 4:46.2, and track 19 at 0:40.9 to 0:43.2, on the door clips (E9).
   The 40 px band and 4-frame confirmation removed most of these, but not all.
6. **A group that waits on the mats before leaving.**
   On `door_0307_1720` about nine people gathered on the mats (the outside of the line) before 0:20 and left through the doors between 0:41 and 0:54.
   They crossed the line long before they left; the doorway region now counts them when they vanish in the doorway, but those counts land at the moment the tracker drops them, not when they actually leave (E9).
7. **Stair traffic.**
   The left end of the door line reaches over the stair landing, so people walking between the downstairs and upstairs flights can be counted as door traffic (E3; noticed by Shokhbek while labelling).

## 7. Scaling to about 50 cameras on one GPU server

This section is a plan, not a measurement: no NVIDIA GPU was available, so only the Mac numbers in section 5 are measured.
Figures from other sources are marked with where they come from.

**The budget.**
At 10 analysed frames per second per camera (D6), 50 cameras need 500 detections per second, plus 500 tracker updates and zone checks.
The event logic is cheap; decoding and the detector are what cost.

**Decoding breaks first.**
H.264 and H.265 frames depend on earlier frames, so a stream must be decoded in full even if only every third frame is analysed: decode cost follows the camera's frame rate and resolution, not the analysis rate.
50 full-HD streams at 25 to 30 fps is beyond what a T4-class card's hardware decoders are reported to handle together with detection (NVIDIA's developer forum reports DeepStream on a T4 falling below 30 FPS past about 30 1080p streams).
So the first lever is the cameras: use each camera's secondary stream (typically 720p or lower) at 10 to 15 fps for analytics and keep the full stream for recording.
Decode on the GPU's NVDEC engines (through GStreamer or NVIDIA DeepStream, or PyNvVideoCodec) so frames stay in GPU memory instead of being copied to the CPU and back.

**Inference: TensorRT, FP16, batched.**
Ultralytics lists YOLO26s at 2.5 ms per 640 px image on a T4 with TensorRT; 500 images per second would keep one T4 more than fully busy at batch 1, while batching raises throughput substantially.
The service already batches the newest frame from every source (D5); at 50 sources the detector would take fixed batches of 16 to 32 with a short deadline.
An L4- or A10-class card, or YOLO26n (1.7 ms on a T4, lower accuracy), gives headroom.
Export with `model.export(format="engine", half=True)`; ONNX Runtime with the TensorRT execution provider is the more portable route.

**The CPU side.**
In this Python service each source is a thread; at 50 sources the GIL and CPU-side resizing become the limit before the GPU does.
The fix is a few worker processes, each owning about a dozen cameras and one TensorRT engine (or all sharing an inference server such as Triton), with resizing and letterboxing done on the GPU.

**Frame rate per camera.**
10 analysed fps is enough for walking people with ByteTrack; queue and counter zones only need about 2 fps, because they are summarised per second, so cameras that watch only a counter can run lower and free capacity for door cameras.

**What breaks, in order.**
1. Decoding full-resolution streams at full frame rate.
2. CPU-side copies, resizing and the Python GIL.
3. GPU inference, once decoding and pre-processing are on the GPU.
4. Operations: 50 reconnect loops, clock drift between cameras, and silent failures, which need per-camera health metrics (frames per second, reconnects, last frame time), alerting on a dead camera, and the frame-dropping policy this service already has so one slow camera cannot back up the others.

## 8. Who did what

Shokhbek chose the direction at each step, labelled both door clips and the first queue marks by hand, and reviews the labels and results.
Claude Code (an AI coding assistant, which the brief allows) wrote the code and documents, ran the experiments, and labelled the remaining queue marks and the expected alerts from extracted frames.
Which labels came from whom is recorded in the label files themselves and in E7.
