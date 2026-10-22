"""CPU-only independent archive, exact reciprocal and bitwise/timing audit.

Does not call the CUDA experiment or its table generator. This is an audit of
fixed saved outputs; the earlier MP240 check is inherited, not rerun here.
"""
from pathlib import Path
from fractions import Fraction
from datetime import datetime, timezone
import hashlib, json, math, re, zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'research/experiments/reciprocal_root_v2'
OLD = ROOT / 'research/experiments/device_root_v1'
OUT = ROOT / 'research/results/reciprocal_root_audit_v1.json'

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def same(a, b):
    assert a.dtype == b.dtype == np.dtype('float64') and a.shape == b.shape
    return bool(np.array_equal(a.view(np.uint64), b.view(np.uint64)))

def main():
    assert not OUT.exists(), 'Preserve the previous audit; use a new version.'
    manifest = read(BASE / 'manifest.json')
    summary = read(BASE / 'summary.json')
    assert manifest['status'] == summary['status'] == 'complete'
    counts = {}
    for key, prefix in [('source_sha256', ROOT), ('artifacts_sha256', BASE), ('generated_sha256', BASE)]:
        for name, digest in manifest[key].items(): assert sha(prefix / name) == digest, name
        counts[key] = len(manifest[key])
    with zipfile.ZipFile(BASE / 'sources.zip') as archive:
        assert set(archive.namelist()) == set(manifest['source_sha256'])
        for name, digest in manifest['source_sha256'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest, name
    counts['archived_sources'] = len(manifest['source_sha256'])

    control = (BASE / 'control_generated/guarded_interval.cuh').read_text()
    variant = (BASE / 'variant_generated/guarded_interval.cuh').read_text()
    # Check literal table actually compiled, rather than trusting proof labels.
    marker = '__device__ __constant__ I integer_reciprocal[52] = {'
    assert variant.count(marker) == 1
    table = variant.split(marker, 1)[1].split('};', 1)[0]
    literals = re.findall(r'\{([^,{}]+),([^,{}]+)\}', table)
    assert len(literals) == 52 and tuple(map(float, literals[0])) == (0., 0.)
    proof = read(BASE / 'reciprocal_proof.json')
    assert len(proof['rows']) == proof['count'] == 51 and proof['rewritten_call_sites'] == 6
    for k, (lo_text, hi_text) in enumerate(literals[1:], 1):
        lo, hi = float.fromhex(lo_text), float.fromhex(hi_text)
        exact = Fraction(1, k)
        assert Fraction(lo) <= exact <= Fraction(hi)
        assert lo == hi or math.nextafter(lo, math.inf) == hi
        row = proof['rows'][k - 1]
        assert row['k'] == row['denominator'] == k and row['numerator'] == 1
        assert (row['lo_hex'], row['hi_hex']) == (lo.hex(), hi.hex())
    # Each replacement uses original interval multiplication; dynamic div stays.
    pattern = (r'#ifdef __CUDACC__\n__device__ __constant__ I integer_reciprocal\[52\] = .*?'
               r'GI_INLINE I div_integer\(I a,int k\)\{return mul\(a,integer_reciprocal\[k\]\);\}\n\n')
    restored, n = re.subn(pattern, '', variant, flags=re.S)
    assert n == 1
    replacements = [
        ('div_integer(term,2*j+1)', 'div(term,pt(2*j+1))'),
        ('div_integer(term,k)', 'div(term,pt(k))'),
        ('div_integer(mul(term,r),j)', 'div(mul(term,r),pt(j))'),
        ('div_integer(mul(term,x),j)', 'div(mul(term,x),pt(j))'),
        ('div_integer(mul(rem,pt(radius)),j)', 'div(mul(rem,pt(radius)),pt(j))'),
        ('div_integer(mul(rem,pt(r)),j)', 'div(mul(rem,pt(r)),pt(j))')]
    for new, old in replacements:
        assert restored.count(new) == 1
        restored = restored.replace(new, old)
    assert restored == control
    for name in ['root.cu', 'binding.cpp', 'guarded_preprocess.cuh', 'guarded_root.cuh']:
        assert (BASE / 'control_generated' / name).read_bytes() == (BASE / 'variant_generated' / name).read_bytes()
        assert (BASE / 'control_generated' / name).read_bytes() == (OLD / 'generated' / name).read_bytes()
    assert control == (OLD / 'generated/guarded_interval.cuh').read_text()

    with np.load(BASE / 'outputs.npz') as data, np.load(OLD / 'outputs.npz') as prior:
        checks = dict(packets_control_variant=same(data['control_packets'], data['variant_packets']),
                      roots_control_variant=same(data['control_roots'], data['variant_roots']),
                      control_packets_original=same(data['control_packets'], prior['packets']),
                      control_roots_original=same(data['control_roots'], prior['roots']))
        roots = data['variant_roots'].copy()
    assert all(checks.values()) and roots.shape == (110, 12)
    finite = roots[:, 0] == 1
    assert finite.sum() == 106 and (roots[:, 0] == 2).sum() == (roots[:, 0] == 3).sum() == 2
    assert (roots[finite, 1] == 0).sum() == 76 and (roots[finite, 1] == 1).sum() == 30
    assert np.isfinite(roots[finite]).all() and np.isnan(roots[~finite, 2:]).all()
    for row in roots[finite]:
        z, lo, hi = map(Fraction, row[2:5])
        assert lo < z < hi and hi - lo <= Fraction(2, 1000)
        assert row[6] < 0 < row[7]  # Strict residual upper left / lower right.
    with np.load(BASE / 'inputs.npz') as data, np.load(OLD / 'inputs.npz') as prior:
        assert data['source'].shape == (7322, 36) and data['sections'].shape == (110, 2)
        for key in ['source', 'sections']: assert np.array_equal(data[key], prior[key])

    pairs = read(BASE / 'paired_timing.json')['rows']
    assert len(pairs) == 5
    for i, row in enumerate(pairs):
        assert row['pair'] == i and row['order'] == (['control', 'variant'] if i % 2 == 0 else ['variant', 'control'])
        for key in ['preprocess', 'root', 'combined']:
            for arm in ['control', 'variant']:
                t = row['results'][arm][key]
                assert t['event_ms'] > 0 and t['wall_ms'] > 0
            ratio = row['results']['variant'][key]['event_ms'] / row['results']['control'][key]['event_ms']
            assert math.isclose(row['ratios'][key], ratio, rel_tol=1e-15)
    ratios = {k: math.exp(sum(math.log(p['ratios'][k]) for p in pairs) / 5)
              for k in ['preprocess', 'root', 'combined']}
    for key, ratio in ratios.items():
        assert math.isclose(ratio, summary['paired_event_ratio_geometric_mean'][key], rel_tol=1e-15)
        for arm in ['control', 'variant']:
            median = float(np.median([p['results'][arm][key]['event_ms'] for p in pairs]))
            assert median == summary['median_event_ms'][arm][key]
    assert summary['preset_combined_ratio_gate'] == .95 and ratios['combined'] <= .95
    assert summary['all_bitwise_checks_pass'] and summary['local_optimization_gate_pass']
    inherited = ROOT / 'research/results/device_root_audit_v1.json'
    a = read(inherited)
    assert a['status'] == 'complete' and not a['failures'] and a['independent_world_dps'] == 240
    failure = read(ROOT / 'research/experiments/reciprocal_root_v1/manifest.json')
    assert failure['status'] == 'failed' and failure['failure_type'] == 'AssertionError'
    result = dict(status='complete', checked_utc=datetime.now(timezone.utc).isoformat(),
        archive_checks=counts, exact_adjacent_reciprocals=51, isolated_replacements=6,
        bitwise_checks=checks, source_rows=7322, finite_models=106, no_root_models=4,
        finite_low=76, finite_high=30, exact_width_and_strict_sign_checks=106,
        timing_pairs=5, paired_event_ratio_geometric_mean=ratios,
        combined_reduction_percent=100*(1-ratios['combined']), speedup_factor=1/ratios['combined'],
        inherited_mp240_audit_sha256=sha(inherited), retained_failure='reciprocal_root_v1',
        local_gate_pass=True,
        scope='Independent CPU archive/exact-rational/bitwise/timing audit. Inherits old MP240 evidence through bitwise-identical outputs; no fresh unseen cohort, full-render/backward/GT or novelty proof.')
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))

if __name__ == '__main__': main()
