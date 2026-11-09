# Experiments

Every test run during the work, in the order it happened, including the ones that exposed problems.
Numbers are as measured; the raw files behind the evaluation runs are in `results/`.
Hardware for all runs: Apple M4 Pro, 24 GB unified memory, macOS 15, detector on the GPU through PyTorch MPS unless stated.

## E1. Finding footage (2026-11-08)

**Question:** which public footage shows people at an entrance and at a counter, from a fixed camera, long enough and under a clear licence?

**What was tried:**

- Pexels and Pixabay search pages for store, queue, cashier, counter and bakery clips.
  Counter and queue clips there run 10 to 19 seconds, for example "busy airport Starbucks with customer queues" (10 s), "busy ice cream shop" (15 s) and "busy supermarket checkout in Auckland" (19 s), and most are handheld or cinematic.
- Wikimedia Commons, the Internet Archive and searches for public queue datasets: no long, clearly licensed footage of people queuing was found.
- CAVIAR, Lisbon shopping centre (CC BY-SA): all 52 clips downloaded (430 MB).
  The reported durations were wrong (MPEG-2 metadata), so frames were counted instead: about 24 minutes per camera view at 384x288 and 25 fps.
  Sample frames showed a fixed view of a shop entrance with 1 to 3 people at a time and no counter.
- MEVA (CC BY 4.0, public S3 bucket): one frame from each of its 28 cameras, then the official activity labels for every clip counted per camera.
  Camera school.G421 is the only one with a counter: a school lobby snack counter with an attendant, 127 labelled purchases, and queues of 3 to 5 people in the 2018-03-07 afternoon.
  Camera school.G420 has the most labelled door crossings (716 entries, 700 exits).

**Result:** MEVA G421 for the counter and G420 for the door (decision D1).

**Problems found on the way:**

- In the two busiest counter clips the attendant was behind the counter in every frame sampled every 15 s, so they cannot exercise the empty-counter alert.
  The 17:35 clip, where the attendant steps away, was added.
- Every G420 clip is one-directional (a wave of arrivals or of departures), so one arrival clip and one departure clip were taken.

## E2. Detector and tracker comparison (2026-11-08)

**Question:** which YOLO model, input size and tracker to use.

**Setup:** 900 frames (30 s) of `counter_1650.avi` and `door_0307_1720.avi`, every frame, Ultralytics 8.4.207, MPS.
"Track IDs" counts distinct IDs; for the same people, more IDs means more broken tracks.
It is a cheap proxy for ID switches, not an accuracy measure, because there were no labels yet.

| Detector, tracker, input | Counter FPS | Counter track IDs | Door FPS | Door track IDs |
|---|---|---|---|---|
| YOLO11n, ByteTrack, 640 | 113 | 76 | 117 | 40 |
| YOLO26n, ByteTrack, 640 | 115 | 57 | 115 | 38 |
| YOLO11s, ByteTrack, 640 | 92 | 60 | | |
| YOLO26s, ByteTrack, 640 | 91 | 53 | 91 | 26 |
| YOLO26n, ByteTrack, 960 | 99 | 59 | | |
| YOLO26s, ByteTrack, 960 | 62 | 39 | | |
| YOLO11n, BoT-SORT, 640 | 67 | 77 | | |
| YOLO26n, BoT-SORT, 640 | 66 | 57 | | |

An earlier probe of YOLO11n on CPU gave 40 FPS against 112 FPS on MPS, and one stream's process peaked at about 790 MB of memory.

**Result:** YOLO26s, ByteTrack, 640 px (decision D2).
YOLO26s at 960 px fragmented least on the counter but at two thirds of the speed; it stays a candidate to check against the labels.

## E3. Placing the zones and the door line (2026-11-08)

**Method:** a coordinate grid drawn over reference frames, first estimates written to `config.yaml`, then every tracked foot point of a whole clip plotted over the zones to check them.

**Findings:**

- The first queue zone was a rectangle in front of the counter.
  The foot-point map showed the queue curving right and back into the room, so the tail of the queue was outside it.
  The zone was reshaped into a five-point polygon that follows the queue but stops short of the seating and the coffee table.
- A whiteboard in the foreground hides the lower bodies of people standing behind it, so their boxes, and therefore their foot points, end at the whiteboard's top edge.
  This shows as a band of foot points along the whiteboard and is a known failure case.
- People at the coffee table to the right of the queue stand close to the zone's right edge; whether they fall inside depends on a few pixels.
- The door line was put on the inner edge of the entrance mats, not on the door threshold.
  People leaving vanish through the doors within a moment, so a line at the threshold would give too few frames on the far side to confirm a crossing (decision D7).
- While labelling, Shokhbek noticed people walking between the downstairs and upstairs flights; the left end of the line reaches over the stair landing, so stair traffic could be counted as door traffic.
  This is to be checked against the hand labels.

## E4. First end-to-end run (2026-11-08)

**Setup:** both sources (counter 16:50, door 7 March 17:20) in one service, offline mode, 10 frames per second analysed per source.

**Result:**

- 108 frames per second in total, 17.7 ms per batch of two frames, mean batch size 2.0, no dropped frames.
- Counter: per-minute queue averages 2.1 to 4.0 with peaks up to 6; two queue alerts, each started and ended once.
- Door: 2 in and 9 out.
  MEVA's own labels list 15 exits for this clip; Shokhbek's later hand count found 13 out and 0 in.
- The staff zone was reported empty for 2 to 8 seconds per minute although the attendant never left.
  The attendant is cut off by the bottom of the frame and partly hidden by the counter, so the detector sometimes loses her.

## E5. Reconnect test over RTSP (2026-11-08)

**Setup:** MediaMTX 1.21.3 on 127.0.0.1 serving the counter clip over RTSP, the door clip as a real-time file, 75 seconds, live JSON endpoint polled during the run.
The RTSP publisher was killed at 25 s and restarted at 40 s.

**Result:**

- At 40 s the endpoint showed the counter disconnected with 1 reconnect while the door had gone from 215 to 366 processed frames.
- The counter reconnected by itself when the stream returned and finished at 6.6 FPS average over the whole run, including the 15 s gap.
- The door ran at 9.8 FPS against a 10 FPS target throughout.

A deprecation warning from Ultralytics on every batch (`half` passed on non-CUDA devices) was fixed by passing FP16 only on CUDA.

## E6. System runs on all evaluation clips (2026-11-08)

`scripts/evaluate.py`, each clip alone, offline, every sampled frame.

| Clip | IN | OUT | Alerts (start to end in s, peak) | FPS |
|---|---|---|---|---|
| door_0309_1010 | 24 | 6 | | 93 |
| door_0307_1720 | 2 | 9 | | 106 |
| counter_1650 | | | queue 0 to 79 (6), queue 254 to 287 (4) | 94 |
| counter_1720 | | | queue 9 to 250 (7) | 90 |
| counter_1735 | | | empty counter 155 to 165 (2) | 91 |

The comparison with the labels follows in E7 and E8.

## E7. Labelling and agreement between labellers (2026-11-08 to 2026-11-09)

**Tools considered:** CVAT, Label Studio and FEVA (a fast event-annotation tool).
The first two are built for boxes and need a server; the task needs only key presses at crossings and a number every 10 s, so a small OpenCV labeller (`scripts/label.py`) was written that draws the same line and zones as the service (decision D9).

**Who labelled what:**

- Shokhbek hand-labelled both door clips with the labeller: 25 crossings in the arrival clip (22 in, 3 out) and 13 in the departure clip (13 out).
- Shokhbek hand-labelled 11 queue marks of `counter_1650` (every 20 s from 10 s to 210 s).
- Claude labelled all 30 marks of `counter_1650` and the marks of `counter_1720` from extracted frames with the zones drawn on, with a written rule: count every person whose feet are inside the queue polygon, judging hidden feet from the body.

**Agreement on the 11 marks both labelled:** exact on 4 of 11; Claude's count is higher on the other 7 by 1 or 2, a mean of +0.82 people.
Looking at the frames, the difference is the rule: Claude counted everyone standing inside the outline, including people standing back from the counter and people at the zone's right edge near the coffee table, while Shokhbek counted the people who were queuing.
This is the same ambiguity the system faces, and it is reported rather than resolved by choosing the labels that suit the system.

## E8. First comparison with the labels (2026-11-09)

`scripts/evaluate.py` with the configuration as committed on 2026-11-08 (YOLO26s, 640 px, 10 fps analysed, zones and line from E3).
Full output: `results/eval/report.json`.

**Door, against Shokhbek's hand counts:**

| Clip | Hand IN | Hand OUT | System IN | System OUT | Error IN | Error OUT | Events matched within 2 s | Extra | Missed |
|---|---|---|---|---|---|---|---|---|---|
| door_0309_1010 (arrivals) | 22 | 3 | 24 | 6 | +2 | +3 | 16 | 14 | 9 |
| door_0307_1720 (departures) | 0 | 13 | 2 | 9 | +2 | -4 | 3 | 8 | 10 |

The totals are close on arrivals, but the event matching is poor: many system crossings do not fall within 2 s of a hand label.
Part of that is timing (the hand label was pressed when the person crossed at the doors, while the system counts at the inner edge of the mats, a second or two later), and part is real error; E9 looks at the individual events.
On the departure clip the system misses 4 of 13 exits; the group of about eight people leaving together is the likely cause.

**Queue, mean absolute error at the 10 s marks:**

| Clip | Labeller | Marks | MAE | Bias (system minus label) | Exact | Mean labelled queue |
|---|---|---|---|---|---|---|
| counter_1650 | Shokhbek | 11 | 1.18 | +1.18 | 3 | 1.73 |
| counter_1650 | Claude | 30 | 0.77 | +0.50 | 12 | 2.40 |
| counter_1720 | Claude | 30 | 0.30 | +0.23 | 23 | 3.63 |
| counter_1735 | Claude | 30 | 0.00 | 0.00 | 30 | 0.47 |

The system never counts fewer people than Shokhbek on his marks; it counts everyone standing in the zone, while he counted people who were queuing.
Against Claude's labels, which follow the geometric rule, the error is lower, and it is lowest on the quiet clip.
The two labellers agree exactly on 4 of their 11 shared marks (E7).

**Alerts, against the expected alerts Claude listed from the 10 s labels:**

| Clip | Kind | Expected | System | Found | Missed | Wrong |
|---|---|---|---|---|---|---|
| counter_1650 | queue | 2 | 2 | 2 | 0 | 1 |
| counter_1720 | queue | 1 | 1 | 1 | 0 | 0 |
| counter_1735 | queue | 0 | 0 | 0 | 0 | 0 |
| counter_1650, 1720 | empty counter | 0 | 0 | 0 | 0 | 0 |
| counter_1735 | empty counter | 0 | 1 | 0 | 0 | 1 |

- On counter_1650 the system's first alert (0 to 79 s) covers both expected alerts (0 to 30 s and 40 to 110 s): the dip at 30 s was bridged, so two expected alerts became one, and it ended about 30 s early.
  Its second alert (254 to 287 s, peak 4) is wrong: the labels show 1 or 2 people then, so the extra people are most likely the ones at the coffee table on the zone edge.
- On counter_1735 the attendant really did leave the counter from about 155 to 165 s (checked every 2 s) and walked to the coffee table and back.
  Nobody was waiting, but the system counted the attendant herself as a waiting customer while she crossed the queue zone, plus a passer-by, and raised an empty-counter alert at 164 s.
  This is the staff versus customer problem the brief lists as a bonus.
