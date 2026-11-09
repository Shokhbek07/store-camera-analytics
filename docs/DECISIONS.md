# Decisions

Each decision lists the alternatives that were checked and why they lost.
Numbers are measured on this project's footage unless a source is cited.

## D1. Footage: MEVA cameras G421 (counter) and G420 (door)

Decided 2026-11-08.

The task needs people passing an entrance and people waiting at a counter or till, from fixed cameras, under a licence that allows download, processing and showing.

**Chosen:** the MEVA dataset (https://mevadata.org), licence CC BY 4.0.
Camera school.G421 looks at a school lobby with a snack counter.
One person serves behind the counter and customers line up in front of it, 3 to 5 deep at peak, with entrance doors in the background.
Camera school.G420 looks down at the building entrance doors.
Both record 1920x1072 at 30 fps in 5-minute clips.

**Alternatives checked:**

- Pexels and Pixabay stock clips.
  Counter and queue clips there run 10 to 19 seconds and are mostly handheld or cinematic.
  The queue measurement needs at least 2 minutes and an alert needs T seconds of waiting, so these are too short.
- CAVIAR, Lisbon shopping centre (CC BY-SA).
  Fixed cameras on a shop entrance, about 24 minutes per view, but 384x288, 1 to 3 people at a time, and no counter or queue.
  Kept as a fallback, not downloaded by the script.
- Wikimedia Commons, the Internet Archive and public queue datasets.
  Searches found no long, clearly licensed footage of people queuing.
- Other MEVA cameras.
  All 28 cameras were sampled and every camera's activity labels were counted.
  G420 has the most labelled door crossings (716 entries, 700 exits), and G421 is the only camera with a counter, with 127 labelled purchases.

**Clip choice within the cameras:**

- Counter: 2018-03-07 16:50, 17:20 and 17:35.
  The first two are the busiest service periods (10 labelled purchases each).
  In both, the staff member was behind the counter in every frame sampled every 15 seconds, so they alone cannot exercise the empty-counter alert.
  The 17:35 clip was added because the staff member steps away during it.
- Door: 2018-03-09 10:10 and 2018-03-07 17:20.
  Door traffic in every G420 clip is one-directional, a wave of arrivals or a wave of departures.
  These two are the strongest arrival clip (41 labelled entries) and departure clip (15 labelled exits), so IN and OUT are both tested.

**Known limits:** the people are actors following a script, the setting is a school lobby rather than a café, and the counter is self-service with one attendant.

## D2. Detector and tracker: YOLO26s with ByteTrack at 640 px

Decided 2026-11-08.

Measured on 900 frames (30 s) of `counter_1650.avi` and `door_0307_1720.avi`, Apple M4 Pro GPU through PyTorch MPS, Ultralytics 8.4.207.
"Track IDs" is the number of distinct IDs the tracker assigned.
For the same people, more IDs means more broken tracks, so it is a cheap proxy for ID switches, not an accuracy measure.

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

**Chosen:** YOLO26s with ByteTrack at 640 px.
It fragments tracks far less than YOLO11n and still runs at three times real time per 30 fps stream.
The detector, input size and tracker are all set in the config, so 960 px can be used where people are small.

**Alternatives:**

- YOLO26s at 960 px fragments least on the counter clip but halves the speed; it stays an option to evaluate against the hand labels.
- BoT-SORT cost about 40% of the speed for no fewer IDs here.
  Its re-identification is off by default, and its camera-motion compensation is wasted on fixed cameras.
- Published comparisons disagree: Ultralytics reports YOLO26n faster and slightly more accurate than YOLO11n on COCO (https://docs.ultralytics.com/compare/yolo26-vs-yolo11), and Roboflow ranks BoT-SORT above ByteTrack on MOT17 but recommends ByteTrack as the starting point (https://trackers.roboflow.com/latest/trackers/comparison/).
  Neither is a person-only, fixed-camera benchmark, which is why the choice rests on the measurement above.

## D3. Downloads: plain curl with SHA-256 checks

Decided 2026-11-08.

`scripts/download.sh` fetches the five clips from MEVA's public S3 bucket and the weights from the Ultralytics release, and refuses any file whose checksum differs.
The AWS CLI was the documented alternative, but it adds a large dependency for five files, and the bucket also serves plain HTTPS.

## D4. Event logic: our own small module, not a library

Decided 2026-11-08.

Door counting, per-second zone occupancy and alerts live in `src/storecam/events.py`, pure Python with unit tests.

- A door crossing counts only when a track's confirmed side flips.
  A side is confirmed when the foot point (bottom centre of the box) is beyond a dead band of `band_px` on either side of the line, within the line's extent, for `min_frames` frames in a row.
  Stopping on the line, stepping onto it and turning back, walking along it, and box jitter therefore never count.
- Zone occupancy is checked on every processed frame and reduced to one value per second: the median queue count and whether staff were present in at least half the frames.
- Each alert opens once its condition has held for `min_seconds` and closes only after the condition has been false for `clear_seconds`, so it cannot flicker; it records start, end and peak.

**Alternative checked:** Roboflow `supervision` (`LineZone`, `PolygonZone`).
`LineZone` only fixed double counting from jittery detections in 0.25 (https://supervision.roboflow.com/0.25.0/changelog/), and `PolygonZone` gives a per-frame count, so dwell time and alerts would still be ours to build (https://github.com/roboflow/supervision/issues/54).
The brief weights this logic at 20% and asks for live changes at the interview, so a small module we own and test is the better trade.

## D5. Multi-stream: reader thread per source, one batched detector

Decided 2026-11-08.

Each source has a reader thread that decodes, paces files at real-time speed, samples `process_fps` frames per second, and keeps only the newest frame in a one-slot mailbox.
One detector loop takes the newest frame from every source, runs them as one batch, and feeds each source's own ByteTrack and analytics.
A slow detector therefore drops frames instead of building a backlog, and a failed source never blocks the others.
On a lost stream the reader reopens it with exponential backoff, and the source's tracker and line state reset, because track IDs do not survive a gap.

**Alternatives checked:**

- Ultralytics' built-in multi-stream loader (`LoadStreams`).
  When any one stream's thread ends it stops every stream, and it reopens a lost stream at once with no backoff, feeding black frames meanwhile.
  That fails the brief's "reconnect without crashing the others".
- One process or one model per stream.
  Simpler isolation, but every stream loads its own model and nothing is batched, which is the opposite of what a 50-camera server needs.

**Evidence (2026-11-08):** the counter clip served over RTSP by MediaMTX on 127.0.0.1 and the door clip as a real-time file, 75 seconds.
The RTSP publisher was killed at 25 s and restarted at 40 s.
The counter reconnected by itself (1 reconnect) while the door stream kept running at 9.8 FPS against a 10 FPS target throughout.
