"""Guarded write of the already syntax-checked research canvas to its existing location."""
from pathlib import Path
import hashlib,json
BASE=Path(__file__).resolve().parent
g=json.loads((BASE/'results/canvas_coefficient_guard_v1.json').read_text(encoding='utf-8'))
s=json.loads((BASE/'results/canvas_coefficient_syntax_v1.json').read_text(encoding='utf-8'))
assert not s['errors'] and s['staged_source_sha256']==g['staged_sha256']
target=Path(g['destination']);assert hashlib.sha256(target.read_bytes()).hexdigest()==g['previous_sha256']
payload=(BASE/'canvas_coefficient_stage.tsx').read_bytes();assert hashlib.sha256(payload).hexdigest()==g['staged_sha256']
target.write_bytes(payload);assert hashlib.sha256(target.read_bytes()).hexdigest()==g['staged_sha256']
print(json.dumps({'updated_existing_canvas':str(target),'sha256':g['staged_sha256']}))
