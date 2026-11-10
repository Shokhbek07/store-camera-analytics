#!/bin/sh
# Speed of the service in the Docker container (CPU), PyTorch against ONNX Runtime (E15).
# Both runs read the two looping RTSP cameras from docker compose in real time, back to back,
# so they share the same machine load; results land in runs/docker/bench_*/run_summary.json
# and each run's log in runs/docker/bench_*.log. The first seconds go to opening the streams.
# Usage: ./scripts/benchmark_docker.sh [seconds per run, default 120]
set -eu
cd "$(dirname "$0")/.."
SECONDS_PER_RUN=${1:-120}
docker compose build storecam
docker compose up -d cameras
[ -f models/yolo26s_dynamic.onnx ] || { echo "export first: uv run storecam export-onnx" >&2; exit 1; }
for w in models/yolo26s.pt models/yolo26s_dynamic.onnx; do
    name=$(basename "$w" | tr . _)
    echo "== $w, $SECONDS_PER_RUN s"
    docker compose run --rm -T storecam run config.yaml \
        --url counter=rtsp://cameras:8554/counter --url door=rtsp://cameras:8554/door \
        --weights "$w" --max-seconds "$SECONDS_PER_RUN" --out "/out/bench_$name" > "runs/docker/bench_$name.log" 2>&1
    python3 -c "
import json, sys
s = json.load(open(sys.argv[1]))
d = s['detector']
if not s['batches']:
    sys.exit('no frames processed; see the log next to ' + sys.argv[1])
print(json.dumps({'weights': d['weights'], 'runtime': d.get('runtime'), 'per_stream_fps': {n: v['fps'] for n, v in s['sources'].items()},
                  'detector_ms_per_batch': s['detector_ms_per_batch'], 'mean_batch': s['mean_batch_size']}))
" "runs/docker/bench_$name/run_summary.json"
done
docker compose stop cameras
