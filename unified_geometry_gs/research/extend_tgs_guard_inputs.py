import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_compositing_guard_v4'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    target=OUT/'input_plan_extended.json';assert not target.exists();old=OUT/'input_plan.json';plan=json.loads(old.read_text());cases=plan['cases'].copy()
    for e in plan['cases']:
        if e['family']!='compositing_cases':continue
        with np.load(ROOT/e['input']) as z:f={k:z[k].copy() for k in z.files}
        n=int(f['n']);p=f['theta'][:9*n].astype(float).reshape(n,3,3);q=(p[...,:2]/(p[...,2:]+1e-7)+1)*128-.5
        edge=np.roll(q,-1,1)-q;length=np.linalg.norm(edge,axis=-1);D=np.linalg.det(np.stack([q[:,1]-q[:,0],q[:,2]-q[:,0]],1));cross=edge[...,0]*(72-q[...,1])-edge[...,1]*(76-q[...,0]);u=(cross*np.sign(D)[:,None]*length.sum(1)[:,None]/(abs(D)[:,None]*length)).min(1)
        op=(np.array(e['desired_alpha'])/u**f['theta'][9*n:10*n]).astype(np.float32);index=3 if 'tail' in e['name'] else 0;shift=e['opacity_ulp_shift']
        for _ in range(abs(shift)):op[index]=np.nextafter(op[index],np.float32(np.inf if shift>0 else -np.inf))
        f['theta'][10*n:11*n]=op;name=e['name']+'_den64';f['name']=name;path=OUT/'inputs'/(name+'.npz');np.savez_compressed(path,**f)
        cases.append(dict(e,name=name,input=path.relative_to(ROOT).as_posix(),sha256=sha(path),paired_original_sha256=e['sha256'],denominator_precision='float64'))
    plan.update(cases=cases,original_plan_sha256=sha(old),addendum_sha256=sha(ROOT/'TGS_COMPOSITING_INPUT_ADDENDUM_V4_1.md'),extension_script_sha256=sha(Path(__file__)))
    target.write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps(dict(cases=len(cases),plan_sha256=sha(target))))
if __name__=='__main__':main()
