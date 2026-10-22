"""Freeze camera-reader runtime files/binaries before new CUDA execution."""
import json
from pathlib import Path
from tgs_split_color_common_v14 import ROOT,sha,write_new
OUT=ROOT/'experiments/tgs_packed_query_v25'
files=[Path(__file__),OUT/'plan.json',ROOT/'experiments/tgs_cross_scene_v10/input_audit.json',
    ROOT/'experiments/tgs_sh_isolation_v12/input_plan.json',ROOT/'experiments/tgs_sh_isolation_v12/identity_audit.json',
    ROOT/'tgs_sh_isolation_common_v12.py',ROOT/'audit_tgs_real_checkpoint.py']
for phase in ['tgs_aux_ablation_v21','tgs_batch_score_v22','tgs_local_image_loss_v24']:
    receipt=ROOT/f'experiments/{phase}/build_receipt.json';b=json.loads(receipt.read_text());p=Path(b['path']);assert sha(p)==b['sha256'];files.extend([receipt,p])
write_new(OUT/'runtime_dependency_addendum.json',dict(inputs={str(p.relative_to(ROOT)):sha(p) for p in files},before_build_and_GPU_diagnostics=True,
    note='Additional explicit camera-reader/runtime provenance; original frozen plan unchanged',training=False))
print('RUNTIME_DEPENDENCIES_FROZEN',flush=True)
