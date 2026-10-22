"""SASS counter audit; runtime zero preflight separately establishes single traversal."""
import json,re,subprocess
from pathlib import Path
from tgs_window_candidates_common import ROOT,OUT,VARIANTS,sha
def main():
    target=OUT/'binary_audit.json';assert not target.exists();folder=OUT/'binary_audit';folder.mkdir();rows=[]
    for v in VARIANTS:
        base=ROOT/'experiments/tgs_phase_cost_v5' if v.startswith('oracle') else OUT;r=json.loads((base/v/'build_receipt.json').read_text());p=Path(r['module_file']);assert sha(p)==r['module_sha256'];s=subprocess.run(['/usr/local/cuda-12.8/bin/cuobjdump','--dump-sass',str(p)],capture_output=True,text=True,timeout=60);assert s.returncode==0;f=folder/(v+'.sass.txt');f.write_text(s.stdout);functions=[]
        for block in s.stdout.split('Function :')[1:]:
            name=block.splitlines()[0].strip()
            if 'renderCUDA' in name and re.search(r'Lb[01]E',name):functions.append(dict(name=name,atomic_add_mnemonics=re.findall(r'\b(?:RED|ATOM)\S*ADD\S*',block)))
        assert functions;count=sum(len(x['atomic_add_mnemonics']) for x in functions);assert (count>0)==r['instrumented'];rows.append(dict(variant=v,module_sha256=r['module_sha256'],SASS_sha256=sha(f),forward_functions=functions,forward_counter_add_instructions=count,instrumented=r['instrumented'],verified=True))
    target.write_text(json.dumps(dict(rows=rows,all_verified=True,script_sha256=sha(Path(__file__)),scope='Static forward template instructions only; no inference of runtime traversal count or latency. Backward gradient atomics retained.'),indent=2)+'\n');print(json.dumps([{k:r[k] for k in ['variant','forward_counter_add_instructions']} for r in rows]),flush=True)
if __name__=='__main__':main()
