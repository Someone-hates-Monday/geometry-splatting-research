#!/usr/bin/env bash
set -eo pipefail
BASE=/mnt/c/Users/Someone/Desktop/SuGaR-main/unified_geometry_gs/output_compare/ggs_trust_rebuild_20261116
echo "=== queue head ==="
head -n 40 "$BASE/queue_105_122.log"
echo "=== queue markers ==="
grep -E 'scan105|scan122|start|complete|=====' "$BASE/queue_105_122.log" | head -n 60
echo "=== scan24 train head/tail ==="
head -n 15 "$BASE/scan24/logs/train.log"
tail -n 8 "$BASE/scan24/logs/train.log"
echo "=== 105 train ==="
head -n 8 "$BASE/scan105/logs/train.log" 2>/dev/null || true
tail -n 5 "$BASE/scan105/logs/train.log" 2>/dev/null || true
echo "=== 122 train ==="
head -n 8 "$BASE/scan122/logs/train.log" 2>/dev/null || true
tail -n 5 "$BASE/scan122/logs/train.log" 2>/dev/null || true
