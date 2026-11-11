#!/bin/sh
# Speed of the service in the Docker container (CPU), comparing variants such as PyTorch against
# ONNX Runtime (E15) or PyTorch thread counts (E21).
# Every run reads the two looping RTSP cameras from docker compose in real time. The variants take
# turns, and the order reverses on every other repeat, so all see the same background load on a
# shared machine. Each run's summary lands in runs/docker/bench_<label>_r<N>/run_summary.json, its
# log next to it, and one JSON line per run in runs/docker/bench.jsonl, with the 1-minute load
# average at the start and the CPU use of every other container at the start and end (100 = one core).
# The first seconds of a run go to opening the streams.
#
# Usage: ./scripts/benchmark_docker.sh [seconds per run, default 120] [repeats, default 1] [LABEL=ARGS ...]
# Each LABEL=ARGS is a variant: extra arguments for `storecam run`. The default compares
#   pytorch="--weights models/yolo26s.pt" onnx="--weights models/yolo26s_dynamic.onnx"
# Example: ./scripts/benchmark_docker.sh 60 2 t8="--cpu-threads 8" t6="--cpu-threads 6"
set -eu
cd "$(dirname "$0")/.."
SECONDS_PER_RUN=${1:-120}
REPEATS=${2:-1}
if [ $# -ge 2 ]; then shift 2; else shift $#; fi
if [ $# -eq 0 ]; then
    [ -f models/yolo26s_dynamic.onnx ] || { echo "export first: uv run storecam export-onnx" >&2; exit 1; }
    set -- "pytorch=--weights models/yolo26s.pt" "onnx=--weights models/yolo26s_dynamic.onnx"
fi

others_cpu() {  # total CPU of containers that are not part of this project, in % of one core
    docker stats --no-stream --format '{{.Name}} {{.CPUPerc}}' \
        | awk '$1 !~ /^store-camera-analytics/ {sub("%", "", $2); s += $2} END {printf "%.0f", s}'
}

run_variant() {  # $1 = LABEL=ARGS, $2 = repeat
    label=${1%%=*}
    args=${1#*=}
    name="${label}_r$2"
    load=$(python3 -c "import os; print(round(os.getloadavg()[0], 1))")
    before=$(others_cpu)
    echo "== $label ($args), repeat $2, $SECONDS_PER_RUN s, load $load, other containers ${before}%"
    # shellcheck disable=SC2086  # ARGS is split into words on purpose
    docker compose run --rm -T storecam run config.yaml \
        --url counter=rtsp://cameras:8554/counter --url door=rtsp://cameras:8554/door \
        $args --max-seconds "$SECONDS_PER_RUN" --out "/out/bench_$name" > "runs/docker/bench_$name.log" 2>&1
    after=$(others_cpu)
    python3 -c "
import json, sys
s = json.load(open(sys.argv[1]))
d = s['detector']
if not s['batches']:
    sys.exit('no frames processed; see the log next to ' + sys.argv[1])
print(json.dumps({'label': sys.argv[2], 'args': sys.argv[3], 'weights': d['weights'], 'runtime': d.get('runtime'),
                  'cpu_threads': d.get('cpu_threads'), 'repeat': int(sys.argv[4]), 'load_1min_start': float(sys.argv[5]),
                  'other_containers_cpu_pct': [int(sys.argv[6]), int(sys.argv[7])],
                  'per_stream_fps': {n: v['fps'] for n, v in s['sources'].items()},
                  'detector_ms_per_batch': s['detector_ms_per_batch'], 'mean_batch': s['mean_batch_size']}))
" "runs/docker/bench_$name/run_summary.json" "$label" "$args" "$2" "$load" "$before" "$after" | tee -a runs/docker/bench.jsonl
}

docker compose build storecam
docker compose up -d cameras
mkdir -p runs/docker
: > runs/docker/bench.jsonl
r=1
while [ "$r" -le "$REPEATS" ]; do
    if [ $((r % 2)) -eq 1 ]; then
        for v in "$@"; do run_variant "$v" "$r"; done
    else  # reverse order
        i=$#
        while [ "$i" -ge 1 ]; do
            eval "v=\${$i}"
            run_variant "$v" "$r"
            i=$((i - 1))
        done
    fi
    r=$((r + 1))
done
docker compose stop cameras
