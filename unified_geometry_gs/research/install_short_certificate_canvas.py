from pathlib import Path
import hashlib,json
BASE=Path(__file__).resolve().parent
g=json.loads((BASE/'results/canvas_short_certificate_guard_v1.json').read_text());s=json.loads((BASE/'results/canvas_short_certificate_syntax_v1.json').read_text());assert not s['errors']
p=BASE/'canvas_short_certificate_stage_v1.tsx';target=Path(g['destination'])
assert hashlib.sha256(p.read_bytes()).hexdigest()==g['staged_sha256'] and hashlib.sha256(target.read_bytes()).hexdigest()==g['previous_sha256']
target.write_bytes(p.read_bytes());assert hashlib.sha256(target.read_bytes()).hexdigest()==g['staged_sha256']
print(json.dumps(dict(status='updated_existing',sha256=g['staged_sha256'],new_tabs_opened=0)))
