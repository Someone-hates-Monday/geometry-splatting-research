"""Hash-guarded update of the existing user research canvas, without opening tabs."""
from pathlib import Path
import hashlib,json
BASE=Path(__file__).resolve().parent
guard=json.loads((BASE/'results/canvas_guarded_guard_v1.json').read_text(encoding='utf-8'))
syntax=json.loads((BASE/'results/canvas_guarded_syntax_v1.json').read_text(encoding='utf-8'));assert not syntax['errors']
stage=BASE/'canvas_guarded_stage_v1.tsx';target=Path(guard['destination'])
assert hashlib.sha256(stage.read_bytes()).hexdigest()==guard['staged_sha256']
assert hashlib.sha256(target.read_bytes()).hexdigest()==guard['previous_sha256']
target.write_bytes(stage.read_bytes())
assert hashlib.sha256(target.read_bytes()).hexdigest()==guard['staged_sha256']
print(json.dumps(dict(status='updated_existing',sha256=guard['staged_sha256'],new_tabs_opened=0)))
