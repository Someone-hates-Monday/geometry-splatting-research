"""Seal all fixed-input v14 outcomes and reader failure, with explicit claim limits."""
import ast,csv,datetime,json
from pathlib import Path
from tgs_split_color_common_v14 import ROOT,OUT,HELPER,sha,write_new

def read(p):return json.loads(p.read_text())

def dependencies(paths):
    found=set(paths);queue=[p for p in found if p.suffix=='.py']
    while queue:
        p=queue.pop()
        for node in ast.walk(ast.parse(p.read_text())):
            names=[node.module] if isinstance(node,ast.ImportFrom) and node.module else [v.name for v in node.names] if isinstance(node,ast.Import) else []
            for name in names:
                q=ROOT/(name.split('.')[0]+'.py')
                if q.is_file() and q not in found:found.add(q);queue.append(q)
    return found

def main():
    target=OUT/'evidence_manifest.json';assert not target.exists()
    plan=read(OUT/'plan.json');result=read(OUT/'results.json');audit=read(OUT/'independent_audit_v2.json');repair=read(OUT/'audit_repair_plan.json')
    assert result['all_isolation_gates'] and result['model_inputs_unchanged'] and result['forwards']==76 and len(result['cases'])==32
    assert audit['all_verified'] and audit['cases']==32 and audit['full_arrays_reconstructed_and_hashed']
    assert not audit['independent_GPU_replay'] and not result['training'] and not result['benchmark']
    assert result['plan_sha256']==audit['plan_sha256']==sha(OUT/'plan.json')
    assert audit['results_sha256']==repair['results_sha256']==sha(OUT/'results.json')
    assert result['script_sha256']==sha(ROOT/'render_tgs_split_color_v14.py')
    assert audit['script_sha256']==repair['driver_sha256']==sha(ROOT/'audit_tgs_split_color_v14_v2.py')
    assert repair['original_script_sha256']==sha(ROOT/'audit_tgs_split_color_v14.py')
    assert repair['frozen_plan_sha256']==sha(OUT/'plan.json') and not repair['rerender']
    assert (OUT/'audit_v1_failure.txt').is_file()
    for n,h in plan['scripts'].items():assert sha(ROOT/n)==h
    assert sha(ROOT/'TGS_SPLIT_COLOR_PROTOCOL_V14.md')==plan['protocol_sha256']
    assert sha(HELPER)==plan['helper_sha256']
    models=[];refs={HELPER}
    for en in plan['models']:
        p=ROOT/en['path'];assert sha(p)==en['sha256'];models.append(dict(**en,bytes=p.stat().st_size))
    for v,en in plan['modules'].items():
        p=Path(en['path']);assert sha(p)==en['sha256'];refs.add(p)
        p=ROOT/en['receipt'];assert sha(p)==en['receipt_sha256'];refs.add(p)
    ca=ROOT/'experiments/tgs_cross_scene_v10/input_audit.json';assert sha(ca)==plan['camera_audit_sha256'];refs.add(ca)
    assert all(all(r['split_noncolor_literal'].values()) for r in result['cases'])
    assert sum(s['appearance_material_count'] for s in audit['summary'])==0
    assert sum(s['G_zero_or_tiny_count'] for s in audit['summary'])==13
    assert sum(s['holding_parent_color_larger_total_RMS_count'] for s in audit['summary'])==9
    fm=read(OUT/'figure_metadata.json');assert fm['script_sha256']==sha(ROOT/'plot_tgs_split_color_v14.py')
    assert fm['results_sha256']==sha(OUT/'results.json') and fm['audit_sha256']==sha(OUT/'independent_audit_v2.json')
    for p,h in fm['files'].items():assert sha(OUT/p)==h
    with (OUT/'split_components.csv').open() as f:assert sum(1 for _ in csv.DictReader(f))==32
    previous=ROOT/'experiments/tgs_sh_shell_v13/evidence_manifest.json'
    expected='43744a895b2a8eab4b983c7eb29a86afb9a312f10aad3bcce0a313381e586c05'
    assert sha(previous)==expected;refs.add(previous)
    write_new(OUT/'figure_visual_review.json',dict(file='split_components.png',sha256=sha(OUT/'split_components.png'),
        all_cases_visible=True,zeros_preserved_at_zero=True,shared_symlog_scale_disclosed=True,
        all_labels_and_legend_visible=True,no_quality_or_training_claim=True))
    write_new(OUT/'research_direction_decision.json',dict(appearance_transport_priority='Lower on current real-camera evidence',
        cases=32,appearance_material_cases=0,G_nonzero_cases=19,zero_cases_retained=13,
        holding_parent_color_larger_total_perturbation=9,
        G_component_mechanisms_separated=False,new_method_or_novelty_established=False,
        GT_quality_gain=False,geometry_gain=False,efficiency_gain=False,training=False,
        scientific_goal_complete=False,
        next='Window redistribution versus scene compositing/sorting diagnostic and direct prior-art comparison; do not expand SH transport training.'))
    names=['TGS_SPLIT_COLOR_PROTOCOL_V14.md','TGS_SPLIT_COLOR_AUDIT_REPAIR_V14.md','TGS_SPLIT_COLOR_RESULT_2026_10_21.md']
    sources=dependencies(list(ROOT.glob('*tgs_split_color*v14*.py'))+[ROOT/n for n in names])
    files=sorted(sources|refs|{p for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts})
    entries=[]
    for i,p in enumerate(files):
        entries.append(dict(path=p.relative_to(ROOT).as_posix(),size=p.stat().st_size,sha256=sha(p)))
        if i%75==74:print(json.dumps(dict(stage='hashing',count=i+1,total=len(files))),flush=True)
    write_new(target,dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=entries,
        source_dependency_files=len(sources),readonly_model_inputs=models,prior_manifest_sha256=expected,
        full_prior_tree_rehashed=False,CPU_reconstruction_audit=True,independent_GPU_replay=False,
        forward_count=76,training=False,scientific_goal_complete=False))
    for en in entries:assert sha(ROOT/en['path'])==en['sha256']
    print(json.dumps(dict(files=len(entries),bytes=sum(e['size'] for e in entries),source_dependency_files=len(sources),
        manifest_sha256=sha(target),scientific_goal_complete=False)),flush=True)

if __name__=='__main__':main()
