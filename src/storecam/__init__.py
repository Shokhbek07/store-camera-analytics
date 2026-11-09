"""storecam: store camera analytics over several video streams."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(prog="storecam")
    sub = p.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run the service on every source in the config")
    run.add_argument("config", nargs="?", default="config.yaml")
    run.add_argument("--out", help="output directory (default: output_dir in the config)")
    run.add_argument("--save-video", action="store_true", help="write annotated, face-blurred video per source")
    run.add_argument("--http", type=int, metavar="PORT", help="serve live counts and open alerts as JSON")
    run.add_argument("--http-host", default="127.0.0.1",
                     help="address for --http (default 127.0.0.1; 0.0.0.0 only inside a container)")
    run.add_argument("--offline", action="store_true",
                     help="process every sampled frame as fast as possible instead of at real-time speed")
    run.add_argument("--max-seconds", type=float, help="stop after this many seconds")
    prev = sub.add_parser("preview", help="draw each source's lines and zones on its first frame")
    prev.add_argument("config", nargs="?", default="config.yaml")
    prev.add_argument("--out", default="runs/preview")
    for sp in (run, prev):
        sp.add_argument("--url", action="append", default=[], metavar="NAME=URL",
                        help="replace a source's url, keeping its zones and lines (repeatable)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from . import config
    cfg = config.load(args.config)
    by_name = {s.name: s for s in cfg.sources}
    for item in args.url:
        name, sep, url = item.partition("=")
        if not sep or name not in by_name:
            p.error(f"--url {item!r}: expected NAME=URL with NAME one of {sorted(by_name)}")
        by_name[name].url = url
    if args.cmd == "run":
        from .service import Service
        if args.offline:
            for s in cfg.sources:
                s.realtime = False
        summary = Service(cfg, Path(args.out) if args.out else None, args.save_video, args.http,
                          args.max_seconds, args.http_host).run()
        brief = {k: v for k, v in summary.items() if k != "sources"}
        brief["sources"] = {n: {k: v for k, v in s.items() if k not in ("crossings", "seconds")}
                            for n, s in summary["sources"].items()}
        print(json.dumps(brief, indent=2))
    elif args.cmd == "preview":
        import cv2

        from . import draw
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for s in cfg.sources:
            cap = cv2.VideoCapture(s.url)
            ok, frame = cap.read()
            cap.release()
            if not ok:
                print(f"{s.name}: cannot read {s.url}")
                continue
            path = out / f"{s.name}.jpg"
            cv2.imwrite(str(path), draw.geometry(frame, s))
            print(path)
