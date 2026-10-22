"""Seal the bounded v12 experiment; never launch CUDA, download, train or overwrite."""
import ast
import datetime
import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'experiments/tgs_sh_isolation_v12'
SOURCE = ROOT / 'sources/tgs_sh_isolation_v12'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(p.read_text())


def write_new(p, value):
    with p.open('x') as f:
        f.write(json.dumps(value, indent=2) + '\n')


def dependencies(paths):
    found = set(paths)
    queue = [p for p in found if p.suffix == '.py']
    while queue:
        p = queue.pop()
        for node in ast.walk(ast.parse(p.read_text())):
            names = ([node.module] if isinstance(node, ast.ImportFrom) and node.module
                     else [n.name for n in node.names] if isinstance(node, ast.Import) else [])
            for name in names:
                q = ROOT / (name.split('.')[0] + '.py')
                if q.is_file() and q not in found:
                    found.add(q)
                    queue.append(q)
    return found


def main():
    target = OUT / 'evidence_manifest.json'
    assert not target.exists()
    plan, frozen, identity, render, quality, audit, rp, math, step = [read(OUT / name) for name in [
        'input_plan.json', 'execution_plan.json', 'identity_audit.json', 'render_results.json',
        'quality_results.json', 'independent_audit.json', 'render_plan.json',
        'math_results.json', 'forward_step_audit.json']]
    assert identity['all_gates'] and identity['quality_allowed'] and len(identity['rows']) == 4
    assert all(r['different_channel_values'] == 0 and r['exact_RGB'] for r in identity['rows'])
    assert audit['all_verified'] and audit['all_isolation_gates']
    assert audit['source_intervention_verified'] and audit['quality_metrics_independently_verified']
    assert audit['noncolor_and_two_negative_controls_independently_verified']
    assert audit['SSIM_direct_patch_channel_checks'] == 9720
    assert len(render['rows']) == 36 and len(quality['rows']) == 20
    assert render['model_inputs_unchanged'] and render['all_isolation_gates']
    assert not render['training'] and not render['timing'] and not quality['training_effect_identified']
    for name, h in frozen['scripts'].items():
        assert sha(ROOT / name) == h
    assert frozen['input_plan_sha256'] == sha(OUT / 'input_plan.json')
    assert frozen['identity_audit_sha256'] == sha(OUT / 'identity_audit.json')
    assert identity['input_plan_sha256'] == sha(OUT / 'input_plan.json')
    assert identity['source_manifest_sha256'] == sha(SOURCE / 'manifest.json')
    for result, script in [(audit, 'audit_tgs_sh_isolation_v12.py'),
                           (quality, 'analyze_tgs_sh_isolation_v12.py')]:
        assert result['script_sha256'] == sha(ROOT / script)
        assert result['render_results_sha256'] == sha(OUT / 'render_results.json')
        assert result['execution_plan_sha256'] == sha(OUT / 'execution_plan.json')
    assert audit['quality_results_sha256'] == sha(OUT / 'quality_results.json')
    assert quality['metrics_sha256'] == sha(ROOT / 'tgs_image_metrics_v11.py')
    assert render['render_plan_sha256'] == sha(OUT / 'render_plan.json')
    assert render['script_sha256'] == sha(ROOT / 'render_tgs_sh_isolation_v12.py')
    assert rp['execution_plan_sha256'] == sha(OUT / 'execution_plan.json')
    assert plan['protocol_sha256'] == sha(ROOT / 'TGS_SH_ISOLATION_PROTOCOL_V12.md')
    assert plan['script_sha256'] == sha(ROOT / 'prepare_tgs_sh_isolation_v12.py')
    refs = set()
    for e in plan['source_edits']:
        for prefix in ['original', 'generated']:
            p = ROOT / e[prefix]
            assert sha(p) == e[prefix + '_sha256']
            refs.add(p)
        a, b = ROOT / e['original'], ROOT / e['generated']
        old = '(glm::vec3)(center_triangle.x, center_triangle.y, center_triangle.z)'
        new = 'glm::vec3(center_triangle.x, center_triangle.y, center_triangle.z)'
        assert a.read_text().count(old) == 1 and b.read_text() == a.read_text().replace(old, new)
    for v, en in rp['modules'].items():
        p = Path(en['path'])
        assert sha(p) == en['sha256']
        refs.add(p)
    for row in render['rows']:
        assert row['all_RGB_finite']
        assert row['v11_old_fields_literal'] is None or all(row['v11_old_fields_literal'].values())
        for key, h in row['files'].items():
            p = OUT / row['scene'] / row['cohort'] / row['mode'] / row['variant'] / (key + '.npy')
            assert sha(p) == h
    assert len(render['comparisons']) == 4
    assert all(r['isolation_gate'] and r['SH3_RGB_changed'] for r in render['comparisons'])
    assert step['all_mid_step_forwards_literal_equal']
    assert step['script_sha256'] == sha(ROOT / 'audit_tgs_sh_forward_steps_v12.py')
    assert step['render_results_sha256'] == sha(OUT / 'render_results.json')
    assert math['exact_checks_pass'] and math['script_sha256'] == sha(ROOT / 'verify_tgs_sh_semantics_math_v12.py')
    glm = ROOT / 'sources/tgs_native_diagnostic_v1/native/third_party/glm/glm/detail/type_vec3.inl'
    assert sha(glm) == math['source_GLM_sha256']
    refs.add(glm)
    acquisition = read(SOURCE / 'manifest.json')
    assert acquisition['total_new_cached_Range_payload_bytes'] <= 96 * 1024 * 1024
    for en in acquisition['files']:
        p = ROOT / en['local']
        assert en['zip_crc_checked'] and sha(p) == en['sha256']
        refs.add(p)
    prior = {}
    for label, folder in [('v10', 'tgs_cross_scene_v10'), ('v11', 'tgs_gt_quality_v11')]:
        p = ROOT / 'experiments' / folder / 'evidence_manifest.json'
        assert sha(p) == plan['old_' + label + '_manifest_sha256']
        prior[label] = dict(path=p.relative_to(ROOT).as_posix(), sha256=sha(p),
                            full_prior_tree_rehash_in_this_seal=False)
        refs.add(p)
    fm = read(OUT / 'figure_metadata_v3.json')
    assert fm['script_sha256'] == sha(ROOT / 'plot_tgs_sh_isolation_v12_v3.py')
    assert fm['quality_sha256'] == sha(OUT / 'quality_results.json')
    assert fm['audit_sha256'] == sha(OUT / 'independent_audit.json')
    for name, h in fm['files'].items():
        assert sha(OUT / name) == h
    write_new(OUT / 'figure_visual_review.json', dict(
        selected='source_effects_v3.png', sha256=sha(OUT / 'source_effects_v3.png'),
        labels_and_ticks_visible=True, different_x_axis_scales_disclosed=True,
        all_four_inputs_retained=True, old_fresh_distinction_visible=True,
        prior_layout_versions_retained=True, inferential_uncertainty_claim=False))
    write_new(OUT / 'research_direction_decision.json', dict(
        four_input_forward_source_intervention_supported=True,
        fresh_views=2, old_posthoc_views=2, negative_controls_passed=True,
        coefficient_only_universal_conversion_counterexample_verified=True,
        complete_native_backward_verified=False, retrained_quality_established=False,
        geometry_quality_evaluated=False, novelty_established=False, training=False,
        scientific_goal_complete=False,
        next='Review appearance-anchor and densification prior art, then specify a distinct function-preserving or bounded-error method; no full training currently authorized.'))
    write_new(OUT / 'runtime_info.json', dict(python=sys.version, platform=platform.platform(),
                                           seal_uses_CUDA=False))
    scripts = list(ROOT.glob('*tgs_sh*v12*.py'))
    docs = [ROOT / n for n in ['TGS_SH_ISOLATION_PROTOCOL_V12.md',
            'TGS_SH_CENTER_MATH_V12.md', 'TGS_SH_ISOLATION_RESULT_2026_10_21.md']]
    deps = dependencies(set(scripts + docs))
    files = sorted(deps | refs | {p for d in [OUT, SOURCE] for p in d.rglob('*')
                                 if p.is_file() and '__pycache__' not in p.parts})
    entries = []
    for i, p in enumerate(files):
        entries.append(dict(path=p.relative_to(ROOT).as_posix(), size=p.stat().st_size, sha256=sha(p)))
        if i % 60 == 59:
            print(json.dumps(dict(stage='hashing', count=i+1, total=len(files))), flush=True)
    write_new(target, dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        files=entries, source_dependency_files=len(deps), prior_manifests=prior,
        all_isolation_and_quality_checks=True, math_counterexample_checked=True,
        forward_count=36, training=False, scientific_goal_complete=False))
    # Re-read the final snapshot so this terminal success asserts all recorded bytes still match.
    for en in entries:
        assert sha(ROOT / en['path']) == en['sha256']
    print(json.dumps(dict(files=len(entries), bytes=sum(e['size'] for e in entries),
                         source_dependency_files=len(deps), manifest_sha256=sha(target),
                         scientific_goal_complete=False)), flush=True)


if __name__ == '__main__':
    main()
