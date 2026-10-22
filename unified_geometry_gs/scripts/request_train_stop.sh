#!/usr/bin/env bash
# Soft-stop a running train.py: finish through the next checkpoint, then exit.
#
# Usage:
#   ./scripts/request_train_stop.sh <model_output_dir>
# Example:
#   ./scripts/request_train_stop.sh ./output_compare/m1_field_occ/scan24
#
# Prefer this over kill -9. Hard abort: kill the python train.py PID.
set -e
if [ -z "${1:-}" ]; then
  echo "Usage: $0 <model_output_dir>" >&2
  exit 1
fi
OUT="$(cd "$1" && pwd)"
FLAG="$OUT/STOP_AT_CHECKPOINT"
mkdir -p "$OUT"
echo "soft-stop requested at $(date -Iseconds)" > "$FLAG"
echo "[soft-stop] wrote $FLAG"
echo "[soft-stop] training will save the next scheduled checkpoint then exit."

# Also nudge via SIGUSR1 if we can find the matching train.py.
if command -v pgrep >/dev/null 2>&1; then
  # Match -m <out> or trailing path fragment.
  while read -r pid; do
    [ -n "$pid" ] || continue
    if kill -USR1 "$pid" 2>/dev/null; then
      echo "[soft-stop] sent SIGUSR1 to pid $pid"
    fi
  done < <(pgrep -f "python.*train.py.*$(basename "$OUT")" || true)
fi
