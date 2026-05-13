#!/bin/bash
# TinyLM — launch training inside a detached tmux session
#
# Wraps `python -m tinylm.train` so SSH disconnects don't kill the run.
# Mirrors stdout to a timestamped log file for offline review.
#
# Usage:
#   bash scripts/run_training.sh configs/run_D_mla_muon.yaml [session_name]
#
# If [session_name] is omitted, it's derived from the config filename
# (e.g. configs/run_D_mla_muon.yaml -> tinylm-run_D_mla_muon).

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <config.yaml> [session_name]" >&2
    exit 2
fi

CONFIG="$1"
SESSION="${2:-tinylm-$(basename "$CONFIG" .yaml)}"

if [ ! -f "$CONFIG" ]; then
    echo "Error: config not found: $CONFIG" >&2
    exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
    echo "Error: tmux is not installed. On Ubuntu: apt-get install -y tmux" >&2
    exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Error: tmux session '$SESSION' already exists." >&2
    echo "Attach with: tmux attach -t $SESSION" >&2
    echo "Or kill it: tmux kill-session -t $SESSION" >&2
    exit 1
fi

mkdir -p logs
LOG="logs/$(date +%Y%m%d-%H%M%S)-${SESSION}.log"

# python -u: unbuffered stdout so `tail -f` and the log stream in real time
tmux new-session -d -s "$SESSION" \
    "python -u -m tinylm.train '$CONFIG' 2>&1 | tee '$LOG'"

echo "Training started in tmux session '$SESSION'."
echo "  Log file:      $LOG"
echo "  Attach:        tmux attach -t $SESSION"
echo "  Detach inside: Ctrl+B then D"
echo "  Tail log:      tail -f $LOG"
echo "  Kill run:      tmux kill-session -t $SESSION"
