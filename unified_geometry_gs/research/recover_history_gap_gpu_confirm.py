"""Recover completed v2 computation after its final console print failed."""
from pathlib import Path
import hashlib,json,shutil,zipfile
import numpy as np
R=Path(__file__).resolve().parent;ROOT=R.parent;OLD=R/'experiments/history_gap_gpu_confirm_v2';OUT=R/'experiments/history_gap_gpu_confirm_v3'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def main():
    assert not OUT.exists();m=read(OLD/'manifest.json');assert m['status']=='failed' and 'int64' in m['failure'];s=read(OLD/'summary.json')
    for n,h in m['source_sha256'].items():assert sha(ROOT/n)==h,n
    for n,h in m['artifacts_sha256'].items():assert sha(OLD/n)==h,n
    with zipfile.ZipFile(OLD/'sources.zip') as z:
        for n,h in m['source_sha256'].items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
    with np.load(OLD/'new_arrays.npz') as d:
        for j,row in enumerate(s['rows']):
            a,b=map(int,d['sections'][j:j+2]);ref=d['reference'][a:b];actual=d['actual'][a:b];rel=float(np.linalg.norm(actual-ref)/max(np.linalg.norm(ref),1e-300));assert rel==row['gradient']['relative_l2'];assert row['loss_error']==abs(float(d['loss'][j])-row['reference_loss'])
    OUT.mkdir(parents=True);shutil.copyfile(OLD/'new_arrays.npz',OUT/'new_arrays.npz');s.update(status='complete',recovered_from='v2 saved computation complete; final console JSON print failed on NumPy int64; no GPU or MP rerun, no numerical changes.');save(OUT/'summary.json',s)
    files=[Path(__file__),OLD/'manifest.json',OLD/'summary.json',OLD/'new_arrays.npz',OLD/'sources.zip',OLD/'failure.txt'];state=dict(status='complete',source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files})
    with zipfile.ZipFile(OUT/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for n in state['source_sha256']:z.write(ROOT/n,n)
    state['artifacts_sha256']={p.name:sha(p) for p in OUT.iterdir() if p.is_file()};save(OUT/'manifest.json',state);print(json.dumps({k:v for k,v in s.items() if k!='rows'}))
if __name__=='__main__':main()
