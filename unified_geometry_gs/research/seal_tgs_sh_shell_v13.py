"""Seal reviewed theory, fixed-field probes, counterevidence and all failures; CPU stdlib only."""
import ast,csv,datetime,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_sh_shell_v13'
FIELDS=OUT/'real_fields_v2';STRESS=OUT/'radius_stress'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def read(p):return json.loads(p.read_text())

def deps(paths):
    result=set(paths);todo=[p for p in result if p.suffix=='.py']
    while todo:
        p=todo.pop()
        for node in ast.walk(ast.parse(p.read_text())):
            names=[node.module] if isinstance(node,ast.ImportFrom) and node.module else [v.name for v in node.names] if isinstance(node,ast.Import) else []
            for name in names:
                q=ROOT/(name.split('.')[0]+'.py')
                if q.is_file() and q not in result:result.add(q);todo.append(q)
    return result

def main():
    target=OUT/'evidence_manifest.json';assert not target.exists()
    plan=read(FIELDS/'plan.json');result=read(FIELDS/'results.json');audit=read(FIELDS/'independent_audit.json')
    assert result['all_defined_checks_pass'] and audit['all_verified'] and audit['cases']==1536
    assert audit['checkpoint_rows_verified'] and audit['maximum_metric_disagreement']<=2e-12
    assert audit['results_sha256']==sha(FIELDS/'results.json') and audit['plan_sha256']==sha(FIELDS/'plan.json')
    assert audit['script_sha256']==sha(ROOT/'audit_tgs_sh_real_fields_v13.py')
    assert result['plan_sha256']==sha(FIELDS/'plan.json')
    assert result['script_sha256']==plan['script_sha256']==sha(ROOT/'probe_tgs_sh_real_fields_v13_v2.py')
    assert plan['original_script_sha256']==sha(ROOT/'probe_tgs_sh_real_fields_v13.py')
    assert plan['protocol_sha256']==sha(ROOT/'TGS_SH_REAL_FIELD_PROTOCOL_V13.md')
    helper=ROOT/'sources/tgs_real_checkpoint_v1/source/utils/sh_utils.py'
    assert plan['helper_sha256']==sha(helper)
    external=[]
    for e in plan['models']:
        p=ROOT/e['path'];assert sha(p)==e['sha256'];external.append(dict(**e,size=p.stat().st_size))
    stress=read(STRESS/'results.json');sa=read(STRESS/'independent_audit.json');sp=read(STRESS/'plan.json')
    assert sa['all_raw_errors_verified'] and sa['cases']==4608 and sa['maximum_RMS_disagreement']<=2e-12
    assert sa['results_sha256']==sha(STRESS/'results.json')
    assert sa['script_sha256']==sha(ROOT/'audit_tgs_sh_radius_v13.py')
    assert stress['script_sha256']==sp['script_sha256']==sha(ROOT/'stress_tgs_sh_radius_v13.py')
    assert stress['plan_sha256']==sha(STRESS/'plan.json')
    assert sp['protocol_sha256']==sha(ROOT/'TGS_SH_RADIUS_STRESS_PROTOCOL_V13.md')
    for name,h in sp['inputs'].items():assert sha(FIELDS/name)==h
    assert len(result['rows'])==1536 and len(stress['rows'])==4608
    assert all(not x['training'] for x in [result,audit,stress,sa])
    four=[g for g in sa['groups'] if g['radius_factor']==4]
    assert sum(g['copy_near_zero'] for g in four)==120
    assert sum(g['worse_than_copy']['L3'] for g in four)==sum(g['worse_than_copy']['L4'] for g in four)==1416
    for file,script in [('math_results.json','verify_tgs_sh_shell_v13.py'),
                        ('radius_asymptotics.json','verify_tgs_sh_radius_asymptotics_v13.py')]:
        m=read(OUT/file);assert m['exact_checks_pass']
        assert m.get('script_sha256',m.get('source_sha256'))==sha(ROOT/script)
    fm=read(STRESS/'figure_metadata.json');assert fm['script_sha256']==sha(ROOT/'plot_tgs_sh_radius_v13.py')
    for p,h in fm['inputs'].items():assert sha(ROOT/p)==h
    for p,h in fm['files'].items():assert sha(STRESS/p)==h
    with (STRESS/'distance_transfer_cases.csv').open() as f:assert sum(1 for _ in csv.DictReader(f))==12288
    with (STRESS/'distance_transfer_summary.csv').open() as f:assert sum(1 for _ in csv.DictReader(f))==64
    prior=ROOT/'experiments/tgs_sh_isolation_v12/evidence_manifest.json'
    prior_sha='91e0ddc12bd940eb774e3c3d2fea1c436e215452799503bcfcbcea8c9c166b3c'
    assert sha(prior)==prior_sha
    oldplan=read(OUT/'real_fields/plan.json')
    assert oldplan['script_sha256']==sha(ROOT/'probe_tgs_sh_real_fields_v13.py')
    assert read(OUT/'real_fields/flowers_selection.json')['ids']==read(FIELDS/'flowers_selection.json')['ids']
    review=dict(figure='radius_stress/distance_transfer.png',sha256=sha(STRESS/'distance_transfer.png'),
        visual_inspection_passed=True,all_four_displacements_visible=True,matched_log2_radius_axes=True,
        no_training_or_image_claim=True,zero_denominators_disclosed=True,original_failures_preserved=True)
    with (OUT/'figure_visual_review.json').open('x') as f:f.write(json.dumps(review,indent=2)+'\n')
    decision=dict(fixed_radius_field_improvement_verified=True,arbitrary_radius_transfer_supported=False,
        four_radius_worse_cases_per_method=1416,total_per_radius=1536,near_zero_copy_cases_per_radius=120,
        full_finite_degree_closure=False,novelty_established=False,real_rendering_gain=False,
        geometry_gain=False,efficiency_gain=False,training=False,scientific_goal_complete=False,
        next='Measure actual-camera color versus coverage components of real geometry splits, and scrutinize radius-conditioned transport versus exact anchor baselines; no full training.')
    with (OUT/'research_direction_decision.json').open('x') as f:f.write(json.dumps(decision,indent=2)+'\n')
    scripts=list(ROOT.glob('*tgs_sh*v13*.py'))
    docs=list(ROOT.glob('TGS_*V13.md'))+[ROOT/'TGS_APPEARANCE_TRANSPORT_RESULT_2026_10_21.md']
    literature=list((ROOT/'literature').glob('appearance_transport_search_v13*.json'))
    sources=deps(scripts+docs)
    files=sorted(sources|set(literature)|{helper,prior}|{p for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts})
    entries=[dict(path=p.relative_to(ROOT).as_posix(),size=p.stat().st_size,sha256=sha(p)) for p in files]
    with target.open('x') as f:f.write(json.dumps(dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        files=entries,source_dependency_files=len(sources),readonly_external_model_inputs=external,
        prior_manifest_sha256=prior_sha,prior_full_tree_rehashed=False,
        all_field_and_raw_stress_audits=True,full_scientific_goal_complete=False,training=False),indent=2)+'\n')
    for e in entries:assert sha(ROOT/e['path'])==e['sha256']
    print(json.dumps(dict(files=len(entries),bytes=sum(e['size'] for e in entries),
        source_dependency_files=len(sources),manifest_sha256=sha(target),training=False,goal_complete=False)),flush=True)

if __name__=='__main__':main()
