#!/bin/sh
# Download the evaluation videos and the detector weights, and check their SHA-256.
# Videos: MEVA dataset (https://mevadata.org), CC BY 4.0, public S3 bucket, no account needed.
# Weights: Ultralytics YOLO26s release asset (AGPL-3.0).
# Files land in data/ and models/, which git ignores. Re-running skips files that verify.
set -eu
cd "$(dirname "$0")/.."
MEVA=https://mevadata-public-01.s3.amazonaws.com/drops-123-r13
WEIGHTS=https://github.com/ultralytics/assets/releases/download/v8.4.0

sha() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -c1-64
    else shasum -a 256 "$1" | cut -c1-64; fi
}

# fetch URL DEST SHA256
fetch() {
    if [ -f "$2" ] && [ "$(sha "$2")" = "$3" ]; then
        echo "ok      $2"
        return
    fi
    mkdir -p "$(dirname "$2")"
    echo "fetch   $2"
    # The MEVA bucket can be slow and drop connections: resume the partial file (-C -), give up on
    # a connection that stalls below 10 kB/s for 60 s, and try again, up to 20 times.
    n=0
    until [ -f "$2.part" ] && [ "$(sha "$2.part")" = "$3" ]; do
        n=$((n + 1))
        if [ "$n" -gt 20 ]; then
            echo "giving up on $2 (checksum mismatch or no connection); delete $2.part to start again" >&2
            exit 1
        fi
        [ "$n" -gt 1 ] && echo "        retry $n, resuming at $(wc -c < "$2.part" 2>/dev/null | tr -d ' ' || echo 0) bytes"
        curl -fsSL -C - --speed-limit 10000 --speed-time 60 -o "$2.part" "$1" || sleep 3
    done
    mv "$2.part" "$2"
}

# Counter camera (school lobby snack counter): busiest service period, plus one with staff absences.
fetch "$MEVA/2018-03-07/16/2018-03-07.16-50-00.16-55-00.school.G421.r13.avi" data/meva/counter_1650.avi \
    88b21714a672c9a9355ef79b311024b5e2dc5ba0ce9635c6d1dc25b53a726786
fetch "$MEVA/2018-03-07/17/2018-03-07.17-20-00.17-25-00.school.G421.r13.avi" data/meva/counter_1720.avi \
    722ab1b4873deda11731e6b4680a2daa2f5fb58a33cf5efe69f72bffc4e01576
fetch "$MEVA/2018-03-07/17/2018-03-07.17-35-00.17-40-00.school.G421.r13.avi" data/meva/counter_1735.avi \
    2f8874ebe55a90783ab918c9cb8e4fd013c2a4260a3e433ca8c8d614de5d3368
# Door camera (school entrance): an arrival wave and a departure wave.
fetch "$MEVA/2018-03-09/10/2018-03-09.10-10-01.10-15-01.school.G420.r13.avi" data/meva/door_0309_1010.avi \
    14fc7531f254dd4f80afecca258a4f41f63b48ce138f115e7f8868cbf3218dc4
fetch "$MEVA/2018-03-07/17/2018-03-07.17-20-01.17-25-01.school.G420.r13.avi" data/meva/door_0307_1720.avi \
    a64cff0bf986cf0aca992c099954cf52db4e1dcb5b6833a53f6a747e638776ce

fetch "$WEIGHTS/yolo26s.pt" models/yolo26s.pt \
    646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b
echo "done"
