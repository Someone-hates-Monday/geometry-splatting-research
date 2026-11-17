#!/usr/bin/env bash
set -eo pipefail
cd /mnt/c/Users/Someone/Desktop/SuGaR-main/unified_geometry_gs
BASE=output_compare/ggs_trust_rebuild_20261116
echo "=== processes ==="
pgrep -af 'train.py|run_jia_rebuild' 2>/dev/null || echo '(none)'
echo "=== scan24 ==="
if [ -f "$BASE/scan24/logs/train_exit.txt" ]; then
  echo "exit:"; cat "$BASE/scan24/logs/train_exit.txt"
fi
if [ -f "$BASE/scan24/logs/train.log" ]; then
  tr '\r' '\n' < "$BASE/scan24/logs/train.log" | grep -E 'Training progress|Saving|\[ITER|error|Error|Traceback' | tail -n 8
fi
ls -la "$BASE/scan24/model/point_cloud" 2>/dev/null || echo 'no point_cloud yet'
for s in scan24 scan105 scan122; do
  ply="$BASE/$s/model/point_cloud/iteration_30000/point_cloud.ply"
  if [ -f "$ply" ]; then
    echo "$s FINAL_PLY $(stat -c%s "$ply") bytes"
  else
    echo "$s no final ply"
  fi
done
echo "=== queue log tail ==="
if [ -f "$BASE/queue_105_122.log" ]; then
  tail -n 20 "$BASE/queue_105_122.log"
else
  echo 'no queue log'
fi
