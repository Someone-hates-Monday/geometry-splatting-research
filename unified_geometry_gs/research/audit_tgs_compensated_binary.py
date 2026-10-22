"""Verify native forward counter removal in SASS, not only a receipt flag."""
import json,re,subprocess
from pathlib import Path
from tgs_compensated_common import ROOT,OUT,VARIANTS,sha
def main():
    target=OUT/'binary_audit.json';assert not target.exists();folder=OUT/'binary_audit';folder.mkdir(exist_ok=True);rows=[]
    for v in VARIANTS:
        r=json.loads((OUT/v/'build_receipt.json').read_text());p=Path(r['module_file']);assert sha(p)==r['module_sha256']
        result=subprocess.run(['/usr/local/cuda-12.8/bin/cuobjdump','--dump-sass',str(p)],capture_output=True,text=True,timeout=60);assert result.returncode==0,result.stderr
        path=folder/(v+'.sass.txt');path.write_text(result.stdout);functions=[]
        for block in result.stdout.split('Function :')[1:]:
            name=block.splitlines()[0].strip()
            if 'renderCUDA' not in name or not re.search(r'Lb[01]E',name):continue
            adds=re.findall(r'\b(?:RED|ATOM)\S*ADD\S*',block)
            maximums=re.findall(r'\b(?:RED|ATOM)\S*MAX\S*',block)
            functions.append(dict(name=name,atomic_add_mnemonics=adds,atomic_max_mnemonics=maximums))
        assert functions,('No forward template functions parsed',v)
        count=sum(len(f['atomic_add_mnemonics']) for f in functions)
        assert (count>0)==r['instrumented'],(v,count,r['instrumented'])
        rows.append(dict(variant=v,module_sha256=r['module_sha256'],SASS_sha256=sha(path),SASS_path=path.relative_to(ROOT).as_posix(),forward_functions=functions,forward_counter_add_instructions=count,instrumented=r['instrumented'],verified=True))
    target.write_text(json.dumps(dict(rows=rows,script_sha256=sha(Path(__file__)),all_verified=True,scope='Forward boolean-template kernels only; backward VJP atomic additions are intentionally retained. Instruction counts are static, not runtime work/latency.',training=False),indent=2)+'\n')
    print(json.dumps([dict(variant=r['variant'],forward_counter_add_instructions=r['forward_counter_add_instructions']) for r in rows]),flush=True)
if __name__=='__main__':main()
