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

## D6. Analyse 10 frames per second per source

Decided 2026-11-08; to be checked against the labels.

The cameras record 30 fps, but people walking in a lobby move only a few pixels between consecutive frames, and ByteTrack keeps identities at much lower rates.
Analysing every third frame cuts detector load by two thirds, which is what makes several streams per GPU possible, and the same setting is the main lever in the 50-camera scaling note.

**Alternatives:** 30 fps (every frame) gives the tracker the easiest job but triples the cost; 5 fps halves it again, but a person crossing the door band in under half a second could then fail the two-frame confirmation (D4).
The rate is `process_fps` in the config.

## D7. Door line on the inner edge of the entrance mats

Decided 2026-11-08.

**Options:**

- A line on the door threshold.
  People leaving pass through the doors and vanish within a moment, so there would be too few frames on the outside of the line to confirm a crossing.
- Counting tracks that start or end in a door region instead of line crossings.
  Robust when the outside is invisible, but the brief asks for line crossings, and track starts and ends are exactly where ID switches happen.
- A line on the inner edge of the mats (chosen).
  Both sides are visible: the mat area in front of the doors is "outside" and the lobby towards the camera and stairs is "inside", each deep enough for the two-frame confirmation.

**Known risk:** people waiting on the mats stand on the "outside" side, and the left end of the line reaches over the stair landing (E3).

## D8. A person's position is the bottom centre of their box

Decided 2026-11-08.

Zones and lines are drawn on the floor, so the point that should be tested is where the person stands, which is the bottom centre of the box for an upright person.
The box centre would put tall people and people near the camera into zones they are only leaning over.

**Known failure:** when the lower body is hidden (the whiteboard in the counter view, the bottom edge of the frame) the box ends early and the foot point is too high (E3).

## D9. Labelling: a small labeller of our own, and who labels what

Decided 2026-11-08, amended 2026-11-09.

**Tool:** CVAT and Label Studio are built for drawing boxes and need a server; FEVA (a fast event-annotation tool) was found but is general-purpose.
The task needs key presses at door crossings and one number every 10 s, so `scripts/label.py` was written in about a hundred lines.
It draws the same line and zones as the service, so hand labels and system output refer to the same geometry.

**Who labels:** the brief asks for hand counts.
Shokhbek labelled both door clips and the first queue marks himself.
To finish in time, Claude labelled the remaining queue marks and the expected alerts from extracted frames, with a written counting rule, in separate files (`labels/*.claude.json`).
Results are reported per labeller, and the agreement between the two on the marks both labelled is reported too (E7), so a reader can see how much the ground truth itself varies.

**Alternative rejected:** using only Claude's labels, or MEVA's own annotations, as ground truth.
Claude's labels come from the same side as the system.
MEVA's annotations are activity labels made for activity recognition ("person enters through structure"), not counts against our door line, and they already differ from the hand count on the departure clip (15 exits labelled by MEVA, 13 by hand).

## D10. Privacy: pixelate the top of every person box

Decided 2026-11-08.

Every published frame or video is written with the top 30% of each person box pixelated (`draw.blur_heads`).

**Alternatives:** a dedicated face detector (for example OpenCV's YuNet) blurs faces more tightly, but misses faces turned away or small, adds a model to download, and in a top-down view faces are often not frontal.
The person boxes already exist, so blurring the head region costs nothing.
**Limit:** a person the detector misses is not blurred, so published media are checked by eye before they are committed.

## D11. Alert thresholds: queue of 3 for 10 s, empty counter for 10 s

Decided 2026-11-08; to be checked against the expected alerts.

The footage has queues of 1 to 7 people in 5-minute clips, so thresholds sized for a real bakery (for example 5 people for 2 minutes) would never fire and the alert logic could not be measured.
A queue alert at 3 or more people for 10 s and an empty-counter alert after 10 s with at least one person waiting give a handful of real events per clip.
`clear_seconds` (5 s for the queue, 3 s for the counter) keeps an alert open through short gaps such as a person stepping out of the zone for a moment.
All of these are per source in the config.

## D12. RTSP test with MediaMTX's own binary, not Docker, on this Mac

Decided 2026-11-08.

Docker Desktop was stopped on this machine, and starting it would also have restarted unrelated containers from another project.
MediaMTX ships a standalone macOS binary, so the reconnect test (E5) ran it directly, listening only on 127.0.0.1.
The one-command Docker setup is still planned; it will run the detector on CPU, because Docker on macOS cannot use the Apple GPU.

## D13. One-command setup: docker compose with MediaMTX cameras, detector on CPU

Decided 2026-11-10.

The brief asks for one config file and one command or docker compose.
`uv run storecam run` already is one command; compose adds a setup that needs only Docker and shows the live-stream path end to end.

**Options considered for the test cameras.**

- Read the files directly in the container: simplest, but it does not exercise RTSP or reconnect, which is the point of a live service.
- MediaMTX plus a separate ffmpeg container publishing into it: works, but two containers and a start-order race.
- **MediaMTX's own `-ffmpeg` image with `runOnInit`, chosen:** one container starts an ffmpeg per path that loops a clip into RTSP and restarts it if it dies.
  The MEVA clips are already H.264, so ffmpeg repackages them with `-c copy` and spends almost no CPU.
  Stopping this container is a ready-made reconnect test.

**How the service gets the stream addresses without a second config.**
Zones and lines must stay in the single `config.yaml`, so `storecam run` gained `--url NAME=URL`, which replaces one source's address and keeps its geometry.
An environment-variable syntax inside the YAML was rejected as hidden magic for one use.

**PyTorch in the image.**
The default Linux PyTorch wheels bundle CUDA libraries of several gigabytes that a CPU container never uses.
The image installs everything else from `uv.lock` and then the CPU build of the same locked PyTorch versions from PyTorch's own index.
A GPU server would use a CUDA base image and the default wheels instead (REPORT section 7).

**Why CPU.**
Docker on macOS runs a Linux VM with no access to the Apple GPU, so in this setup the detector runs on CPU.
In the container this reaches about 5 to 7 FPS per stream against the 10 FPS target (E12, E15), so compose is the easy, portable demo and the native GPU run is the one that meets the target on this Mac.

**Network.**
Every published port is bound to 127.0.0.1, and the cameras are local files, so nothing leaves the machine.
The status endpoint listens on 0.0.0.0 only inside the container (`--http-host`), published to the host's 127.0.0.1.

## D14. Staff versus customer: uniform colour plus time in the staff zone

Decided 2026-11-10, from E14.

The brief offers this as a bonus ("for example by uniform colour or apron") and it addresses a measured error: the attendant counted as a waiting customer.

**Options considered.**

- Tracking history only (a track that has been behind the counter is staff): fails here because the attendant's track breaks often and customers at the end of the counter step into the staff zone (E14).
- Uniform colour only: separates well per box, but customers in similar light clothes were left out of the queue more often than the attendant (E14).
- A trained staff/customer classifier or a re-identification embedding: needs labelled crops for each store, adds a model to run per person, and a person embedding moves towards recognising individuals, which the brief rules out.
- **Uniform colour plus time in the staff zone, chosen:** both are cheap, explainable, and configured per camera; together they removed every wrong exclusion seen on two clips while keeping the attendant leaving the counter.

**How it works.**
`src/storecam/staff.py`: the median Lab colour of the torso part of each box is compared with `staff_appearance.lab`; a track is staff when most of its last 20 colours match and it has spent `min_zone_seconds` in the staff zone.
Staff are left out of the queue count; whether someone is in the staff zone is still decided by anyone standing there, as the brief defines it.
The reference colour comes from `scripts/calibrate_staff.py`, which measures people in the staff zone, so no colour is picked by hand.
Removing `staff_appearance` from the config turns it off.

**Limits** are listed in E14; the main one is that it inherits every ID switch of the tracker.

## D15. Exported detector: ONNX Runtime, with CoreML on the Mac GPU

Decided 2026-11-11, from E15.

The brief offers "TensorRT or ONNX Runtime export, with FPS and accuracy against PyTorch" as a bonus.

**Options considered.**

- TensorRT: needs an NVIDIA GPU, which this machine does not have; it stays the plan for the 50-camera server (REPORT section 7).
- A native CoreML export (`.mlpackage`): runs only on Apple hardware and is not ONNX Runtime; ONNX Runtime's CoreML execution provider reaches the same Apple GPU from the portable ONNX file.
- An MLX port of YOLO26: its authors report 1.1 to 2.6 times PyTorch MPS on an M4 Pro, but it is a separate port of the model rather than an export, and the numbers are the vendor's own; not tried.
- OpenVINO: aimed at Intel hardware; this machine and the Docker VM are ARM; not tried.
- **ONNX Runtime, chosen:** one exported file runs on any CPU, on the Apple GPU through the CoreML execution provider, and on NVIDIA through the CUDA or TensorRT providers.

**How it works.**
`storecam export-onnx` (`src/storecam/export.py`, first written as `scripts/export_onnx.py`) writes two exports with Ultralytics: one with any batch and image size, and one fixed at 384x640 (a 16:9 frame at 640 wide, padded as PyTorch pads it) and batch 2.
Both keep the head and the outside NMS that the PyTorch run uses, so detections can match exactly.
Setting `model.weights` to an `.onnx` file is the only change; `Detector` then runs it through ONNX Runtime and `run_summary.json` records the runtime and execution provider.
On the Mac GPU, Ultralytics asks CoreML for its older NeuralNetwork format, which computed in FP16, changed a few boxes and was slower (E15); `Detector` rebuilds that session as an FP32 MLProgram on CPU and GPU.
This reaches into one Ultralytics internal (the predictor's backend session), which is acceptable because the Ultralytics version is pinned; if a later version moves it, the service fails at start-up instead of running on a different setup silently.

**Default.**
`config.yaml` keeps the PyTorch weights: the ONNX export gave identical results, and with two streams on the Mac GPU the whole service ran about 17% faster with it, but both runtimes keep up with two real-time streams there, the export is slower for a single stream (the fixed batch-2 export pads it), and it adds an export step to the setup (E15).
Docker image (CPU only):
In the container the export was 1.2 to 3.4 times faster than PyTorch while other projects loaded the machine, the ratio growing with the load, but on a quiet machine PyTorch was about 1.4 times faster (119 against 168 ms per batch, medians of three alternating pairs, E15).
Shokhbek decided on 2026-11-11 to switch only if a run on a quiet machine confirmed a large gain; it did not, so the Docker image keeps PyTorch (settled 2026-11-11).
On a server shared with other work, the export's steadiness under load (168 to 491 ms per batch across all container runs, against PyTorch's 119 to 1153 ms) would be a reason to choose it; `--weights models/yolo26s_dynamic.onnx` switches a container run without rebuilding.

## D16. Live page: a static page on the existing endpoint

Decided 2026-11-11.

The brief's last bonus is "a tiny web page or API endpoint showing live counts".
The service already served live counts as JSON (D5).

**Options considered.**

- A dashboard framework (Streamlit, Gradio): another dependency and process for a few numbers.
- Server push over WebSocket or server-sent events: more code on both sides, while a one-second refresh matches the one-second zone sampling.
- A live video view: not needed to read counts, and it would show unblurred people from the cameras.
- **One self-contained page served by the same small HTTP server, chosen:** `/` returns the page, which reads `/status.json` once a second and shows one card per camera (in, out, queue, staff at the counter, connection, open alerts with how long they have been open); no external files, so it also works offline.

The page builds its content with text nodes only, so source names and values from the config cannot inject markup.
The server still binds to 127.0.0.1 unless `--http-host` says otherwise (D13).

## D17. Who counts in the queue: a zone edge in front of the chairs, and one box per person

Decided 2026-11-11, from E16.

The last wrong queue alert (`counter_1650`, 4:14 to 4:47) came from a man seated at a table behind the queue, whose foot point lay on the zone's top edge, not from the coffee table as first assumed (E16).

**Options considered.**

- Count only people who have been in the zone for some seconds: matched Shokhbek's stricter counting better (MAE 1.18 to 0.73 at 3 s) but was worse against Claude's marks on the other two clips and did not remove the alert, because the seated man stayed for half a minute.
- Count only people standing still: also kept the seated man, and walking customers are part of a moving queue.
- A queue as a chain of people starting at the counter: more principled for long queues, but needs spacing thresholds that change with perspective; not tried in the time left.
- A separate "ignore" polygon over the seating: the same effect as moving the zone edge, with one more config field.
- **Move the zone's top edge to just in front of the chairs nearest the counter, chosen:** zones are drawn per camera at set-up by looking at the scene, and the config already said the edge should stop short of the seating; it did not.
  The edge was placed from the scene (the nearest chair legs reach about y 715 on `counter_1720`), and every edge from y 700 to 750 removed the alert, so the result does not hang on a few pixels.
- **Also leave duplicate boxes out of the queue count, chosen:** the detector sometimes adds an upper-body box inside a person's full box, which non-maximum suppression keeps because the two overlap too little, so one person counted twice.
  A box at least 85% inside a larger box that reaches lower is treated as part of it (`duplicate_boxes` in `src/storecam/events.py`); this improved every labeller's error on every clip.
  It is applied to the queue count only, so door counting is unchanged.

**Limits.**
Both were chosen on the clips they are measured on; the edge is justified by the furniture and the duplicate rule by what it removes, but there is no held-out counter clip.
Chairs move, so a zone drawn once can drift out of date and should be checked when the furniture changes.
A person almost completely hidden behind another, with the visible part inside the front person's box, would be treated as a duplicate.

## D18. A person counts in the queue after 2 s in the queue zone

Decided 2026-11-11, from E19, after Shokhbek asked for a rule closer to his counting.

**Options considered** (all replayed on the saved tracks, E19).

- Count everyone in the zone (as before): matches Claude's labelling rule, but counts people walking through or just arriving, which the brief's "people waiting" does not mean.
- Count people standing still: closer to Claude's marks than the time rules, but no better than 0.73 on Shokhbek's, and it drops people moving up in the line.
- Count people after 3, 5 or 8 s in the zone: no better than 2 s on Shokhbek's marks and worse on Claude's, as longer waits miss people who just joined.
- A queue as a chain of people from the counter: needs spacing thresholds that change with perspective; not tried.
- **Count a person after 2 s in the queue zone, chosen:** the smallest wait with the lowest error on Shokhbek's marks (1.00 to 0.55), passing a bar set before the results were seen, with every expected alert still found and none wrong.

**How it works.**
`Dwell` in `src/storecam/events.py` records when each track's foot point entered the queue zone and leaves the track out of the count until it has been there `zones.queue_min_dwell_s` seconds (2 in `config.yaml` for the counter; 0 counts everyone).
Up to 1 s outside the zone keeps the time, so a foot point that jumps for a frame does not restart it.

**Cost.**
A queue alert starts about 2 s later, and a new arrival appears in the live queue count 2 s after reaching the zone.
Against Claude's marks, which count people walking in, the error rose (0.37, 0.17 and 0.00 to 0.53, 0.37 and 0.07).
The 2 s is fitted to 11 marks from one clip (E19, Limits).
On 19 marks Shokhbek counted blind afterwards, not used to choose it, the rule still helped, by less: 0.58 people per mark against 0.68 without it (E22).
It also undercounts for the first 2 s after a start or reconnect, when nobody has yet been in view for 2 s.
