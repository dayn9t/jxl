#!/usr/bin/env bash
# Resumable rsync of dataset_v2 to sgcc0 with keepalive + auto-retry.
# Broken-pipe deaths observed on this link (see ds-relay memory); retry until complete.
set -u
SRC=/home/jiang/ws/trash/cabin/dataset_v2/
DST=sgcc0:ws/trash/cabin/dataset_v2/
LOG=/home/jiang/ws/trash/rsync_v2.log

for attempt in $(seq 1 20); do
    echo "=== attempt $attempt $(date +%H:%M:%S) ===" >>"$LOG"
    rsync -aL --partial --timeout=120 \
        -e "ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=6 -o TCPKeepAlive=yes" \
        --info=stats1 "$SRC" "$DST" >>"$LOG" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
        echo "=== rsync OK after attempt $attempt ===" >>"$LOG"
        exit 0
    fi
    echo "=== attempt $attempt failed rc=$rc, retrying in 10s ===" >>"$LOG"
    sleep 10
done
echo "=== rsync FAILED after 20 attempts ===" >>"$LOG"
exit 1
