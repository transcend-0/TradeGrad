#!/usr/bin/env bash
# Train TradeTGD on one task (2018-2023, six yearly windows, worst-30% CVaR), then score every
# checkpoint on the 2024-2025 test window. Edit the YOUR_* placeholders and settings below.
#
# Launched detached (nohup, backgrounded): this returns immediately, training + scoring continue
# in the background and survive the terminal disconnecting, and everything lands in the log file
# printed below.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

export AKQUANT_DATA_ROOT=YOUR_DATA_ROOT
export OPENROUTER_API_KEY=YOUR_API_KEY
PY=python

TASK=cn_cross           # cn_cross | cn_time | us_cross | us_time
STEPS=100
WORST_RATIO=0.3

# TradeTGD hyperparameters
SAMPLE_SIZE=3
LARGE_REVISION_PROB=0.7
MEMORY_WINDOW=5
VERBOSE=0                # 1 to write per-step traces/gradients/memory summaries to disk

WORKERS=24                # parallel workers for the test-window scoring pass

if [ "$AKQUANT_DATA_ROOT" = "YOUR_DATA_ROOT" ]; then
    echo "Set AKQUANT_DATA_ROOT (edit this script) first." >&2
    exit 1
fi
if [ "$OPENROUTER_API_KEY" = "YOUR_API_KEY" ]; then
    echo "Set OPENROUTER_API_KEY (edit this script) first." >&2
    exit 1
fi

verbose_flag=""
[ "$VERBOSE" = "1" ] && verbose_flag="--verbose"

mkdir -p "log/${TASK}"
log="log/${TASK}/run.log"
nohup bash -c "
    '$PY' -m src.run_one --task '$TASK' --max-steps '$STEPS' --worst-ratio '$WORST_RATIO' \
        --sample-size '$SAMPLE_SIZE' --large-revision-prob '$LARGE_REVISION_PROB' \
        --memory-window '$MEMORY_WINDOW' $verbose_flag &&
    '$PY' -m src.eval_runs --task '$TASK' --workers '$WORKERS'
" > "$log" 2>&1 &
disown
echo "started in the background (pid $!) -> $log"
