#!/usr/bin/env bash
set -eo pipefail
ROOT=/mnt/c/Users/Someone/Desktop/SuGaR-main
BASE="$ROOT/unified_geometry_gs/output_compare/ggs_trust_rebuild_20261116"
echo "=== tree ==="
ls -la "$BASE"
for s in scan24 scan105 scan122; do
  echo "=== $s ==="
  ls -la "$BASE/$s" || true
  if [[ -d "$BASE/$s/model/point_cloud/iteration_30000" ]]; then
    ls -la "$BASE/$s/model/point_cloud/iteration_30000"
    python3 - <<PY
import hashlib, pathlib
p=pathlib.Path("$BASE/$s/model/point_cloud/iteration_30000/point_cloud.ply")
h=hashlib.sha256(p.read_bytes()).hexdigest()
print("ply_bytes", p.stat().st_size, "sha256", h)
PY
  fi
done
echo "=== control_trunc4 ==="
ls -laR "$BASE/scan24/control_trunc4" | head -60
echo "=== control results.json ==="
if [[ -f "$BASE/scan24/control_trunc4/vis/results.json" ]]; then
  cat "$BASE/scan24/control_trunc4/vis/results.json"
elif [[ -f "$BASE/scan24/results.json" ]]; then
  echo "WARNING: results at scan24 root"
  cat "$BASE/scan24/results.json"
else
  echo "NO results.json"
fi
echo
echo "=== smoke ==="
cat "$BASE/scan24/smoke/smoke_receipt.json"
echo
echo "=== sha smoke assets ==="
python3 - <<'PY'
import hashlib, pathlib
base=pathlib.Path("/mnt/c/Users/Someone/Desktop/SuGaR-main/unified_geometry_gs/output_compare/ggs_trust_rebuild_20261116/scan24/smoke")
for name in ["smoke_receipt.json","smoke_arrays.npz","smoke_redmask.png"]:
  p=base/name
  print(name, p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest())
PY
echo "=== impl_diff exists ==="
ls -la "$BASE/scan24/logs/impl_diff.patch" 2>/dev/null || echo missing
echo "=== train exit markers ==="
tail -n 5 "$BASE/scan24/logs/train.log" 2>/dev/null || true
tail -n 3 "$BASE/queue_105_122.log" 2>/dev/null || true
