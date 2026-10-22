"""Read-only compatibility repair for NumPy2.4 size-one scalars, separate audit output."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
original=root/'audit_tgs_split_color_v14.py'
out=root/'experiments/tgs_split_color_v14'
source=original.read_text()
assert source.count("float(inp['sigma'])")==1 and source.count("float(inp['opacity'])")==1
plan=dict(original_script_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
    driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    results_sha256=hashlib.sha256((out/'results.json').read_bytes()).hexdigest(),
    frozen_plan_sha256=hashlib.sha256((out/'plan.json').read_bytes()).hexdigest(),
    changes=['sigma/opacity size-one .item()','separate audit output filename'],rerender=False)
with (out/'audit_repair_plan.json').open('x') as f:f.write(json.dumps(plan,indent=2)+'\n')
source=source.replace("float(inp['sigma'])","float(inp['sigma'].item())")
source=source.replace("float(inp['opacity'])","float(inp['opacity'].item())")
source=source.replace("OUT/'independent_audit.json'","OUT/'independent_audit_v2.json'")
exec(compile(source,str(Path(__file__)),'exec'))
