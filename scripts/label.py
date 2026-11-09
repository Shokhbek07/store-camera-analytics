"""Hand labelling for the evaluation: door crossings, queue counts, and expected alerts.

The window shows the video with the same door line and zones the service uses (from the
config), so hand labels and system output refer to the same geometry. Labels are saved after
every change, to labels/<mode>_<clip>.json. The window shows faces unblurred; it is local only.

Usage:
    uv run python scripts/label.py door    [SOURCE] [VIDEO]   # press i / o as people cross the line
    uv run python scripts/label.py queue   [SOURCE] [VIDEO]   # type the queue count every 10 s
    uv run python scripts/label.py alerts  [SOURCE] [VIDEO]   # mark when expected alerts start and end

SOURCE is a source name in config.yaml; it defaults to door for door mode and counter otherwise.
VIDEO defaults to that source's url.

Keys (all modes):
    space  play / pause          a / d  back / forward 2 s      , / .  one frame back / forward
    [ / ]  slower / faster       u      undo the last label      q / Esc  save and quit
Door mode:    i  someone crossed IN       o  someone crossed OUT
Alerts mode:  1  queue alert starts/ends  2  empty-counter alert starts/ends
Queue mode:   jumps to every 10 s mark and pauses; type the number of people whose feet are in the
              queue zone, then Enter. s toggles "staff present". Backspace edits. Play with space
              to look at motion; it stops again at the next mark.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from storecam import config, draw  # noqa: E402

STEP_S = 10


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("door", "queue", "alerts"):
        print(__doc__)
        sys.exit(1)
    mode, rest = sys.argv[1], sys.argv[2:]
    cfg = config.load("config.yaml")
    names = {s.name for s in cfg.sources}
    source_name = rest.pop(0) if rest and rest[0] in names else ("door" if mode == "door" else "counter")
    src = next(s for s in cfg.sources if s.name == source_name)
    video = rest[0] if rest else src.url
    out = Path("labels") / f"{mode}_{Path(video).stem}.json"
    out.parent.mkdir(exist_ok=True)
    data = json.loads(out.read_text()) if out.exists() else {"video": video, "mode": mode, "labels": []}
    labels: list[dict] = data["labels"]

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pos, playing, speed = 0, False, 1.0
    typed, staff = "", True
    open_alert: dict[str, float] = {}
    for lab in labels:  # restore open alerts from a previous session
        if mode == "alerts" and lab.get("end") is None:
            open_alert[lab["kind"]] = lab["start"]

    def save() -> None:
        out.write_text(json.dumps(data, indent=1))

    def frame_at(i: int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, img = cap.read()
        return img if ok else None

    def next_mark(t: float) -> float:
        done = {lab["t"] for lab in labels}
        m = 0.0
        while m in done or m < t - 1e-6:
            m += STEP_S
        return m

    if mode == "queue":
        pos = int(next_mark(0) * fps)
    cv2.namedWindow("label", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("label", 1440, 804)
    img = frame_at(pos)
    last = time.monotonic()
    while True:
        if img is None:
            playing = False
            pos = max(0, min(pos, n_frames - 1))
            img = frame_at(pos)
            if img is None:
                break
        t = pos / fps
        view = draw.geometry(img.copy(), src)
        hud = f"{mode}  t={t:7.2f}s / {n_frames / fps:.0f}s  x{speed:g}  {'PLAY' if playing else 'PAUSE'}"
        if mode == "door":
            n_in = sum(1 for lab in labels if lab["dir"] == "in")
            hud += f"   IN={n_in} OUT={len(labels) - n_in}   [i]n [o]ut [u]ndo"
        elif mode == "queue":
            hud += f"   labelled {len(labels)}   queue: {typed or '_'}  staff={'yes' if staff else 'NO'}  (Enter saves)"
        else:
            hud += f"   alerts {len(labels)}  open: {', '.join(open_alert) or '-'}   [1] queue  [2] empty counter"
        cv2.rectangle(view, (0, 0), (view.shape[1], 56), (0, 0, 0), -1)
        cv2.putText(view, hud, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imshow("label", view)

        delay = max(1, int(1000 / (fps * speed) - (time.monotonic() - last) * 1000)) if playing else 30
        key = cv2.waitKey(delay) & 0xFF
        last = time.monotonic()

        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            playing = not playing
        elif key == ord("a"):
            pos = max(0, pos - int(2 * fps))
            img = frame_at(pos)
        elif key == ord("d"):
            pos = min(n_frames - 1, pos + int(2 * fps))
            img = frame_at(pos)
        elif key == ord(","):
            pos = max(0, pos - 1)
            img = frame_at(pos)
        elif key == ord("."):
            pos = min(n_frames - 1, pos + 1)
            img = frame_at(pos)
        elif key == ord("["):
            speed = max(0.25, speed / 2)
        elif key == ord("]"):
            speed = min(8.0, speed * 2)
        elif key == ord("u") and labels:
            removed = labels.pop()
            if mode == "alerts":
                open_alert.pop(removed["kind"], None)
            save()
        elif mode == "door" and key in (ord("i"), ord("o")):
            labels.append({"t": round(t, 2), "dir": "in" if key == ord("i") else "out"})
            save()
        elif mode == "alerts" and key in (ord("1"), ord("2")):
            kind = "queue" if key == ord("1") else "empty_counter"
            if kind in open_alert:
                for lab in labels:
                    if lab["kind"] == kind and lab.get("end") is None:
                        lab["end"] = round(t, 2)
                del open_alert[kind]
            else:
                labels.append({"kind": kind, "start": round(t, 2), "end": None})
                open_alert[kind] = t
            save()
        elif mode == "queue":
            if ord("0") <= key <= ord("9"):
                typed += chr(key)
            elif key in (8, 127):
                typed = typed[:-1]
            elif key == ord("s"):
                staff = not staff
            elif key in (13, 10) and typed and abs(t - round(t / STEP_S) * STEP_S) > 0.2:
                pos = int(round(t / STEP_S) * STEP_S * fps)  # off the mark: go back to it first
                img, playing = frame_at(pos), False
            elif key in (13, 10) and typed:
                mark = round(round(t / STEP_S) * STEP_S, 2)
                labels[:] = [lab for lab in labels if lab["t"] != mark]
                labels.append({"t": mark, "queue": int(typed), "staff_present": staff})
                labels.sort(key=lambda lab: lab["t"])
                save()
                typed, staff = "", True
                nm = next_mark(mark)
                pos = int(nm * fps)
                if pos >= n_frames:
                    print("every 10 s mark is labelled")
                    break
                img = frame_at(pos)
                playing = False

        if playing:
            step = max(1, int(round(speed))) if speed >= 1 else 1
            pos += step
            if mode == "queue":
                nm = next_mark(t + 1e-3)
                if pos >= int(nm * fps):
                    pos, playing = int(nm * fps), False
            img = frame_at(pos) if step > 1 else (cap.read()[1] if pos < n_frames else None)
    save()
    cv2.destroyAllWindows()
    print(f"saved {len(labels)} labels to {out}")


if __name__ == "__main__":
    main()
