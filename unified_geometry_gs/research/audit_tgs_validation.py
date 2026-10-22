"""Recheck saved evidence without CUDA, training, SciPy, or Torch.

Does not replay GPU execution. Checks recorded outputs against independent CPU
reference and strict interval certificate, and packages the exact two-line fix.
"""
import difflib
import hashlib
import json
from pathlib import Path
import numpy as np
from verify_tgs_perspective_certificate import verify
from tgs_native_reference import render

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiments/tgs_subdivision_v1'


def main():
    source=ROOT/'sources/tgs_native_diagnostic_v1'
    manifest=json.loads((source/'manifest.json').read_text())
    for item in manifest['files']:
        assert hashlib.sha256((source/item['path']).read_bytes()).hexdigest()==item['sha256']
    cert=json.loads((OUT/'perspective_certificate.json').read_text())
    verified=verify(cert)
    recorded=json.loads((OUT/'perspective_verified_with_l1_bound.json').read_text())
    assert verified==recorded
    math_result=json.loads((OUT/'math_results.json').read_text())
    assert math_result['passed'] and math_result['symbolic']['opacity_feasibility_threshold']=='31/34'
    cuda=json.loads((OUT/'cuda_results.json').read_text())
    assert cuda['passed'] and len(cuda['fixtures'])==18
    max_difference=0.
    for row in cuda['fixtures']:
        raw=np.load(OUT/(row['name']+'.npz'))
        reference,radii,details=render(raw['points'],raw['sigma'],raw['opacity'],
                                      raw['view'].astype(float),raw['projection'].astype(float))
        assert np.array_equal(reference,raw['reference'])
        assert np.isfinite(raw['native']).all()
        assert np.array_equal(raw['native'][0],raw['native'][1])
        assert np.array_equal(raw['native'][0],raw['native'][2])
        error=float(np.abs(raw['native'][0]-reference).max())
        assert abs(error-row['max_abs_reference_difference'])<1e-14
        assert error <= cuda['calibration']['reference_abs_tolerance']
        assert radii.tolist()==row['reference_radii']==row['native_radii']
        max_difference=max(max_difference,error)
    shdir=OUT/'sh_direction_v1'
    before=json.loads((shdir/'results.json').read_text())
    after=json.loads((shdir/'fixed_results.json').read_text())
    assert before['confirmed'] and after['confirmed'] and after['fixed_variant']
    for result,fixed in [(before,False),(after,True)]:
        assert len(result['cases'])==3
        for case in result['cases']:
            raw=np.load(shdir/(case['case']+('_fixed' if fixed else '')+'.npz'))
            assert np.allclose(raw['points'][0].astype(float).mean(0),case['center'],rtol=0,atol=1e-15)
            observed=np.median(raw['native_sh'][:,raw['ratio_mask']]/raw['white'][0,raw['ratio_mask']],axis=1)
            assert np.array_equal(observed,np.array(case['observed_rgb'],dtype=np.float32))
            expected=np.array(case['true_center_rgb'] if fixed else case['comma_expression_rgb'])
            assert np.max(np.abs(observed-expected))<2e-6
            dsh=np.array(case['dSH']); analytic=np.array(case['analytic_dSH'])
            error=float(np.max(np.abs(dsh-analytic))/np.max(np.abs(analytic)))
            assert abs(error-case['sh_gradient_relative_error'])<1e-15 and error<2e-4
    patch=[]
    old='(glm::vec3)(center_triangle.x, center_triangle.y, center_triangle.z)'
    new='glm::vec3(center_triangle.x, center_triangle.y, center_triangle.z)'
    for name in ['forward.cu','backward.cu']:
        original=(source/'native/cuda_rasterizer'/name).read_text()
        fixed=(shdir/'fixed_src'/name).read_text()
        assert original.count(old)==1 and fixed==original.replace(old,new)
        patch.extend(difflib.unified_diff(original.splitlines(True),fixed.splitlines(True),
                      fromfile='a/cuda_rasterizer/'+name,tofile='b/cuda_rasterizer/'+name))
    (shdir/'sh_direction_fix.patch').write_text(''.join(patch))
    paths=list(OUT.glob('*.json'))+list(OUT.glob('*.npz'))+list(shdir.glob('*.json'))+list(shdir.glob('*.npz'))
    paths += [ROOT/n for n in ['verify_tgs_moments.py','verify_tgs_perspective_certificate.py',
              'prepare_tgs_native_diagnostic.py','tgs_native_diagnostic_backend.py',
              'tgs_native_reference.py','run_tgs_native_diagnostic.py','probe_tgs_sh_direction.py',
              'plot_tgs_subdivision_diagnostic.py','audit_tgs_validation.py',
              'TGS_VALIDATION_RESULT_2026_10_20.md','TGS_REAL_EVENT_PROTOCOL.md']]
    paths += [shdir/'sh_direction_fix.patch',OUT/'subdivision_diagnostic.png',OUT/'subdivision_diagnostic.svg']
    # Audit output itself is excluded to avoid recursive self-hashing.
    paths=[p for p in paths if p.name!='evidence_audit.json']
    report={'passed':True,'source_files_verified':len(manifest['files']),
            'saved_cuda_cases_recomputed':18,'cuda_reexecuted':False,
            'max_abs_native_reference_error':max_difference,
            'perspective_certificate_passed':True,
            'relative_continuous_L1_floor':verified['display_L1_lower_bound'],
            'SH_cases_per_variant':3,'SH_forward_and_coefficient_gradient_records_verified':True,
            'geometry_gradients_verified':False,'real_data_improvement_verified':False,
            'scope':'saved artifact audit; not independent GPU execution or peer review',
            'files':[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                     for p in sorted(set(paths))]}
    (OUT/'evidence_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='files'}))


if __name__=='__main__':
    main()
