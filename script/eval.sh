#!/usr/bin/env bash
# Score an already-trained task's checkpoints on the fixed 2024-2025 test window, without
# retraining. Edit the YOUR_* placeholder and settings below.
#
# Launched detached (nohup, backgrounded): this returns immediately, scoring continues in the
# background and survives the terminal disconnecting.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

export AKQUANT_DATA_ROOT=YOUR_DATA_ROOT
PY=python

TASK=cn_cross           # cn_cross | cn_time | us_cross | us_time
WORKERS=24

if [ "$AKQUANT_DATA_ROOT" = "YOUR_DATA_ROOT" ]; then
    echo "Set AKQUANT_DATA_ROOT (edit this script) first." >&2
    exit 1
fi

mkdir -p "log/${TASK}"
log="log/${TASK}/eval.log"
nohup "$PY" -m src.eval_runs --task "$TASK" --workers "$WORKERS" > "$log" 2>&1 &
disown
echo "started in the background (pid $!) -> $log"
