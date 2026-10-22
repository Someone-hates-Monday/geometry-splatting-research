"""Read-only verification of the WSL binaries/generated sources and raw models."""
from pathlib import Path
import argparse,hashlib,json
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise ValueError('Fresh output required')
    m=read(a.run/'manifest.json');assert m['status']=='complete';checked_generated=0;original={};binaries=[];ply=[]
    for variant,record in m['binaries'].items():
        assert sha(Path(record['path']))==record['sha256'];binaries.append(variant)
        for name,digest in record['metadata']['original_source_sha256'].items():
            if name in original:assert original[name]==digest
            original[name]=digest
    for name,digest in original.items():assert sha(ROOT/name)==digest,name
    for variant,files in m['generated_sha256'].items():
        for name,digest in files.items():assert sha(Path(name))==digest,name;checked_generated+=1
    for scan in (37,55,63):
        r=read(ROOT/f'research/results/median_cpu_scan{scan}.json')
        expected=next(v for k,v in r['source_sha256'].items() if k.endswith('point_cloud.ply'))
        p=ROOT/f'output_compare/ggs_pipeline/scan{scan}/point_cloud/iteration_30000/point_cloud.ply'
        assert sha(p)==expected;ply.append({'scan':scan,'sha256':expected})
    result={'status':'complete','checked_utc':datetime.now(timezone.utc).isoformat(),'binaries_verified':binaries,
        'generated_files_verified':checked_generated,'unique_original_source_files_verified':len(original),'raw_ply_verified':ply,
        'source_sha256':sha(Path(__file__)),'limits':['Asset hashes only; no new GPU execution or root/gradient oracle.']}
    a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='raw_ply_verified'},indent=2))
if __name__=='__main__':main()
