from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,zipfile
import numpy as np
from plyfile import PlyData
R=Path(__file__).resolve().parent;ROOT=R.parent;BASE=R/'experiments/history_gap_adapter_v3'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def metrics(a,b):
    rel=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300));blocks={name:float(np.linalg.norm(a[:,i:j]-b[:,i:j])/max(np.linalg.norm(b[:,i:j]),1e-30)) for name,i,j in [('xyz',0,3),('log_scale',3,6),('quaternion',6,10),('opacity_logit',10,11)]};return rel,blocks
def main():
    out=R/'results/history_gap_adapter_audit_v1.json';assert not out.exists();m=read(BASE/'manifest.json');s=read(BASE/'summary.json');assert m['status']==s['status']=='complete' and s['gate_pass']
    for name,h in m['source_sha256'].items():assert sha(ROOT/name)==h,name
    for name,h in m['artifacts_sha256'].items():assert sha(BASE/name)==h,name
    with zipfile.ZipFile(BASE/'sources.zip') as z:
        for name,h in m['source_sha256'].items():
            if not name.endswith('.ply'):assert hashlib.sha256(z.read(name)).hexdigest()==h,name
    for value in m['binaries'].values():
        path=Path(value.get('path',value.get('binary_relative_path','')));path=path if path.is_absolute() else ROOT/path;assert sha(path)==value['sha256']
    reports=[];external=[]
    for scene in s['scenes']:
        scan=scene['scan'];stem=f'scan{scan}_0000';assert scene['gate_pass'] and scene['RNG_unchanged'] and scene['optimizer_binding_verified'] and scene['parameter_snapshot_restored'] and scene['other_parameters_gradients_and_Adam_finite'] and scene['saturated_opacity_gradient_exactly_zero']
        with np.load(BASE/(stem+'_replay.npz')) as d:
            for name in ['rgb','alpha','normal','radii','depth']:
                for arm in ['query','csr']:assert np.array_equal(d['public_'+name].view(np.uint32),d[arm+'_'+name].view(np.uint32))
            assert np.array_equal(d['query_counters'].view(np.uint32),d['csr_counters'].view(np.uint32))
        with np.load(BASE/(stem+'_scatter.npz')) as d:
            old=R/'experiments/history_gap_raw_v1'/(stem+'_gradients.npz')
            with np.load(old) as ref:assert np.array_equal(d['reference_per_node'],ref['reference']) and np.array_equal(d['ids'],ref['primitive_ids']) and np.array_equal(d['sections'],ref['sections']) and np.array_equal(d['pixels'],ref['pixels'])
            with np.load(R/'experiments/world_root_vjp_v1'/(stem+'_gradients.npz')) as world:assert np.array_equal(d['source'].view(np.uint64),world['source'].view(np.uint64)) and np.array_equal(d['raw'].view(np.uint64),world['raw'].view(np.uint64))
            unique,inverse=np.unique(d['ids'],return_inverse=True);expected=np.zeros((len(unique),11));np.add.at(expected,inverse,d['reference_per_node']/len(d['pixels']));assert np.array_equal(expected,d['reference']) and np.array_equal(unique,d['unique_ids']);relative,blocks=metrics(d['actual'],expected);assert relative==scene['scatter']['relative_l2'] and blocks==scene['scatter']['blocks'] and relative<=1e-5 and max(blocks.values())<=1e-4
        with np.load(BASE/(stem+'_sample.npz')) as d:
            assert len(d['pixels'])==scene['sample_count'] and int(np.count_nonzero(d['counts']==0))==scene['sample_empty']==int(d['empty'][0]);assert np.diff(d['sections']).sum()==len(d['ids']);expected_loss=(float(d['values'].sum())+scene['sample_empty']*np.log(4)**2)/scene['sample_count'];assert abs(expected_loss-float(d['loss'][0]))<1e-14
        ply=ROOT/f'output_compare/ggs_pipeline/scan{scan}/point_cloud/iteration_30000/point_cloud.ply';data=PlyData.read(ply).elements[0].data;opacity=np.asarray(data['opacity']);ids=np.flatnonzero(np.isposinf(opacity));assert not np.isnan(opacity).any() and not np.isneginf(opacity).any()
        with np.load(BASE/(stem+'_opaque.npz')) as d:assert np.array_equal(d['ids'],ids) and len(ids)==scene['preserved_positive_inf_opacity']
        for key,value in scene['zero_gradient_difference'].items():assert value is not None and value<=max(1e-5,5*scene['control_repeat_gradient_difference'][key])
        for key,value in scene['zero_parameter_difference'].items():assert value<=max(1e-5,5*scene['control_repeat_parameter_difference'][key])
        arms={a:[r for r in scene['steps'] if r['arm']==a] for a in ['disabled_a','disabled_b','computed_zero','positive']};assert all(len(v)==3 and [r['step'] for r in v]==[0,1,2] for v in arms.values())
        with np.load(BASE/(stem+'_disabled_a_selected.npz')) as d:baseline_ids=d['ids'];baseline=d['raw'];mask=np.isposinf(baseline)
        for arm in ['disabled_b','computed_zero','positive']:
            with np.load(BASE/(stem+'_'+arm+'_selected.npz')) as d:assert np.array_equal(d['ids'],baseline_ids) and np.array_equal(np.isposinf(d['raw']),mask) and np.isfinite(d['raw'][~mask]).all()
        for path in [ROOT/f'output_compare/ggs_pipeline/scan{scan}/cfg_args',ROOT.parent/f'dtu/DTU/scan{scan}/images/0000.png',ROOT.parent/f'dtu/DTU/scan{scan}/sparse/0/cameras.bin',ROOT.parent/f'dtu/DTU/scan{scan}/sparse/0/images.bin']:
            assert path.exists() and path.stat().st_mtime<m['created_epoch'];external.append(dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size,postrun_checked_unchanged_mtime_precedes_run=True))
        reports.append(dict(scan=scan,scatter_relative_l2=relative,scatter_max_block=max(blocks.values()),sample_count=scene['sample_count'],sample_empty=scene['sample_empty'],opaque_preserved=len(ids),zero_gradient_max=max(scene['zero_gradient_difference'].values()),control_repeat_gradient_max=max(scene['control_repeat_gradient_difference'].values()),zero_parameter_max=max(scene['zero_parameter_difference'].values()),disabled_repeat_parameter_max=max(scene['control_repeat_parameter_difference'].values()),wall_medians_ms={arm:float(np.median([r['total_diagnostic_wall_ms'] for r in rows])) for arm,rows in arms.items()},positive_capture_median_ms=float(np.median([r['capture']['capture_total_wall_ms'] for r in arms['positive']])),positive_prepare_median_ms=float(np.median([r['prepare_loss_wall_ms'] for r in arms['positive']])),positive_regularizer=[r['regularizer'] for r in arms['positive']],positive_rgb=[r['rgb_loss'] for r in arms['positive']],disabled_rgb=[r['rgb_loss'] for r in arms['disabled_a']]))
    empty=read(R/'results/history_gap_adapter_empty_v1.json');assert empty['gate_pass'] and empty['all_empty_gradient_exactly_zero'];assert empty['mixed_loss_error']<1e-14 and empty['mixed_gradient_relative_error']<1e-5
    for name,h in empty['source_sha256'].items():assert sha(ROOT/name)==h
    import diff_gaussian_rasterization
    public=Path(diff_gaussian_rasterization._C.__file__);assert public.stat().st_mtime<m['created_epoch'];external.append(dict(path=str(public),sha256=sha(public),postrun_checked_unchanged_mtime_precedes_run=True))
    result=dict(status='complete',audit_gate_pass=True,checked_utc=datetime.now(timezone.utc).isoformat(),source_checks=len(m['source_sha256']),artifact_checks=len(m['artifacts_sha256']),model_files_excluded_from_source_zip_but_hash_verified=True,scenes=reports,optimizer_steps=36,live_empty_case=1,mixed_empty_fixture_gate=True,external_inputs_postrun_hashes=external,limits=['Fresh-Adam RGB-only fixed-camera/topology memory smoke, all49-trained input PLY; not original full training, heldout evidence or GT improvement.','Existing positive-inf logits preserved as fixed opaque coordinates with exact zero gradients; not all raw parameters finite.','Zero-control passes engineering bounds based on measured atomic repeat noise, not bit/statistical equivalence.','Update wall times include finite/binding checks and sync but exclude artifact transfers/writes and state restoration.'])
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='external_inputs_postrun_hashes'}))
if __name__=='__main__':main()
