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
Some of it may be timing (a key press lagging the crossing, or the system confirming a side only after two frames beyond the dead band), and some is real error; E9 looks at the individual events to separate the two.
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
  Nobody was waiting, but the system counted the attendant as a waiting customer while crossing the queue zone, plus a passer-by, and raised an empty-counter alert at 164 s.
  This is the staff versus customer problem the brief lists as a bonus.

## E9. Why the door counts were off, and what fixed them (2026-11-10)

**Looking at the events.**
Listing the hand and system crossings side by side for both door clips showed three separate problems:

1. Timing: on the arrival clip the system's crossings come 1 to 2.5 s before the matching key press, which fits the reaction time of pressing a key while watching.
   With a 2 s matching window, correct crossings were scored as both "extra" and "missed"; from here on matching is reported at 2 s and 3 s.
2. Flip-flops: one track counted out and in again within half a second (for example track 216 at 285.7 s and 286.2 s, track 19 at 40.9 s, 41.4 s and 43.2 s), when the door or another person cut off part of the box and the foot point jumped across the 20 px band.
3. The departing group: frames every 2 s showed the group of about nine gathering on the mats before 20 s, which is already the outside of the line (D7), and leaving through the doors between 41 s and 54 s.
   They had crossed the line long before they left, so their departure crossed nothing.

**Variants tried** (`scripts/sweep_door.py`, both door clips, same hand counts; full output `results/sweep_door.json`).
"Count error" is the sum over both clips of |system IN minus hand IN| plus |system OUT minus hand OUT|.

| Variant | Count error | Arrivals IN/OUT (hand 22/3) | Departures IN/OUT (hand 0/13) | Matched at 3 s, arrivals | Matched at 3 s, departures |
|---|---|---|---|---|---|
| baseline (mats line, band 20, 2 frames) | 11 | 24/6 | 2/9 | 19 (+11, -6) | 3 (+8, -10) |
| band 40 | 9 | 21/4 | 0/6 | 19 (+6, -6) | 2 (+4, -11) |
| 4 frames | 8 | 22/5 | 1/8 | 20 (+7, -5) | 3 (+6, -10) |
| band 40, 4 frames | 11 | 21/4 | 0/4 | 22 (+3, -3) | 2 (+2, -11) |
| 960 px input | 8 | 22/5 | 1/8 | 19 (+8, -6) | 3 (+6, -10) |
| 960 px, band 40, 4 frames | 9 | 22/4 | 0/5 | 19 (+7, -6) | 2 (+3, -11) |
| 15 fps, band 40, 6 frames | 12 | 21/4 | 0/3 | 22 (+3, -3) | 2 (+1, -11) |
| line on the door threshold | 22 | 6/2 | 5/13 | 7 (+1, -18) | 6 (+12, -7) |
| threshold, band 10 | 19 | 11/3 | 6/15 | 9 (+5, -16) | 7 (+14, -6) |
| threshold, 3 frames | 23 | 5/2 | 5/13 | 6 (+1, -19) | 6 (+12, -7) |
| threshold, 960 px | 20 | 9/3 | 5/15 | 8 (+4, -17) | 8 (+12, -5) |
| mats line + doorway region | 8 | 24/6 | 2/12 | 19 (+11, -6) | 4 (+10, -9) |
| **mats line + doorway region, band 40, 4 frames** | **4** | **21/4** | **0/11** | **22 (+3, -3)** | **7 (+4, -6)** |
| mats line + doorway region, 4 frames | 4 | 22/5 | 1/12 | 20 (+7, -5) | 5 (+8, -8) |
| mats line + doorway region, 960 px | 5 | 22/5 | 1/11 | 19 (+8, -6) | 5 (+7, -8) |

**What the table says.**

- A wider band and more frames remove most flip-flops (extra arrivals events fall from 11 to 3) but do nothing for the departing group.
- Moving the line to the door threshold fixes departures (13 out against 13) and breaks arrivals (6 in against 22): an arriving person is first seen in the doorway, already on the line, so never has a confirmed outside side.
  A single line cannot serve both directions on this camera.
- The 960 px input changes little here; the misses are not small or faint people.
- The fix that works is to keep the mats line and add a doorway region (`door_region`, `LineCounter` in `src/storecam/events.py`, with six new unit tests): a track first seen in the doorway starts on the outside, and a track that vanishes in the doorway without ever having been counted is counted OUT when the tracker drops it.
  With the wider band and four frames it cuts the count error from 11 to 4 and the unmatched events on arrivals from 17 to 6.

**Adopted:** mats line, doorway region, band 40 px, 4 frames, now in `config.yaml`.

**Caveat:** this was chosen on the same two clips it is measured on, so these numbers are optimistic.
A check on two door clips not used for tuning follows in E10.

## E10. The new door settings on two clips not used for tuning (2026-11-10)

E9 chose its settings on the same two clips it measured, so they were checked on two other G420 clips, chosen before running anything: the busiest arrival clip left in MEVA's labels (2018-03-13 16:30) and a clip with traffic both ways (2018-03-12 10:00).
Neither was hand-labelled by us, so the reference is MEVA's own annotations ("person enters/exits through structure"), which are activity labels with their own definition.
On 2018-03-12 10:00 the two MEVA annotation teams disagree with each other (4 in and 4 out against 7 in and 2 out), which shows how rough this reference is.
Results file: `results/heldout_door.json`; the clips are in `data/meva/heldout/` and are not part of `scripts/download.sh`.

| Clip | MEVA labels, IN/OUT | Before (E8 settings) | After (E9 settings) | Count error before | Count error after |
|---|---|---|---|---|---|
| 2018-03-13 16:30 | 40/3 (UMD-IBM) | 44/3 | 41/3 | 4 | 1 |
| 2018-03-12 10:00 | 4/4 (UMD-IBM) | 10/4 | 7/4 | 6 | 3 |
| 2018-03-12 10:00 | 7/2 (CMU) | 10/4 | 7/4 | 5 | 2 |

The new settings are closer to the reference on both unseen clips, against either annotation team, so the E9 improvement is not only an artefact of tuning on the test clips.
With references this rough, that is the claim the data supports, and no more.

**Full evaluation rerun with the E9 settings (2026-11-10).**
Door against the hand counts, events matched within 2 s: arrivals 21 in and 4 out against 22 and 3 (19 matched, 6 extra, 6 missed); departures 0 in and 11 out against 0 and 13 (5 matched, 6 extra, 8 missed).
The counter results came out identical to E8, number for number, which also confirms that offline runs are deterministic.

## E11a. Speed measured while other work was loading the machine (2026-11-10)

`scripts/benchmark.py` ran while Docker Desktop was running another project's containers: four frontend CI containers and a backend, using about 8 of the 14 CPU cores and 6.4 GB of memory (load average 17).
These numbers are kept because they show how much shared load matters, not as the service's speed; the clean run is E11b.
File: `results/benchmark_under_docker_load.json`.

| Run | Total FPS | Per stream | Detector ms per batch | GPU memory |
|---|---|---|---|---|
| 2 streams, real time (target 10 per stream) | 19.6 | 9.8, 9.8 | 19.2 | 1197 MB |
| 1 stream, as fast as possible, MPS | 47.7 | 47.7 | 19.7 | 1214 MB |
| 2 streams, as fast as possible, MPS | 73.6 | 36.8, 36.8 | 25.6 | 1214 MB |
| 4 streams, as fast as possible, MPS | 92.6 | 23.2 each | 40.4 | 1231 MB |
| 1 stream, CPU | 7.2 | 7.2 | 137 | |
| 2 streams, CPU | 4.4 | 2.2 each | 450 | |
| 4 streams, CPU | 8.1 | 2.0 each | 484 | |

For comparison, E2 on a quiet machine measured YOLO26s at 91 FPS for one stream on MPS, and E4 measured 108 FPS for two streams; under this load they fell to 48 and 74.
Even under load the real-time run held 9.8 of the 10 FPS target per stream.
