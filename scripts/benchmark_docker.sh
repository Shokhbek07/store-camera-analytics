#!/bin/sh
# Speed of the service in the Docker container (CPU), PyTorch against ONNX Runtime (E15).
# Every run reads the two looping RTSP cameras from docker compose in real time. The two runtimes
# take turns, and the order alternates between repeats, so both see the same background load on a
# shared machine. Each run's summary lands in runs/docker/bench_<weights>_r<N>/run_summary.json,
# its log next to it, and one JSON line per run (with the 1-minute load average at the start)
# in runs/docker/bench.jsonl. The first seconds of a run go to opening the streams.
# Usage: ./scripts/benchmark_docker.sh [seconds per run, default 120] [repeats, default 1]
set -eu
cd "$(dirname "$0")/.."
SECONDS_PER_RUN=${1:-120}
REPEATS=${2:-1}
PT=models/yolo26s.pt
ONNX=models/yolo26s_dynamic.onnx
[ -f "$ONNX" ] || { echo "export first: uv run storecam export-onnx" >&2; exit 1; }
docker compose build storecam
docker compose up -d cameras
mkdir -p runs/docker
: > runs/docker/bench.jsonl
r=1
while [ "$r" -le "$REPEATS" ]; do
    if [ $((r % 2)) -eq 1 ]; then order="$PT $ONNX"; else order="$ONNX $PT"; fi
    for w in $order; do
        name="$(basename "$w" | tr . _)_r$r"
        load=$(python3 -c "import os; print(round(os.getloadavg()[0], 1))")
        echo "== $w, repeat $r, $SECONDS_PER_RUN s, load $load"
        docker compose run --rm -T storecam run config.yaml \
            --url counter=rtsp://cameras:8554/counter --url door=rtsp://cameras:8554/door \
            --weights "$w" --max-seconds "$SECONDS_PER_RUN" --out "/out/bench_$name" > "runs/docker/bench_$name.log" 2>&1
        python3 -c "
import json, sys
s = json.load(open(sys.argv[1]))
d = s['detector']
if not s['batches']:
    sys.exit('no frames processed; see the log next to ' + sys.argv[1])
print(json.dumps({'weights': d['weights'], 'runtime': d.get('runtime'), 'repeat': int(sys.argv[2]),
                  'load_1min_start': float(sys.argv[3]), 'per_stream_fps': {n: v['fps'] for n, v in s['sources'].items()},
                  'detector_ms_per_batch': s['detector_ms_per_batch'], 'mean_batch': s['mean_batch_size']}))
" "runs/docker/bench_$name/run_summary.json" "$r" "$load" | tee -a runs/docker/bench.jsonl
    done
    r=$((r + 1))
done
docker compose stop cameras
