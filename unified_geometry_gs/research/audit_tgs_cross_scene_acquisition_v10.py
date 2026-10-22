"""Account for successful public Range payloads across one documented terminal retry."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'sources/tgs_cross_scene_v10'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    target=OUT/'acquisition_audit.json';assert not target.exists();manifest=json.loads((OUT/'manifest.json').read_text());assert len(manifest['files'])==10;total=0;ranges=[]
    for p in sorted((OUT/'ranges').glob('*.json')):
        q=json.loads(p.read_text());raw=p.with_suffix('.bin');assert raw.stat().st_size==q['bytes']==q['end']-q['start']+1 and sha(raw)==q['sha256'];assert q['status']==206 and q['total']==15832586976 and q['bytes']<=16*1024*1024;total+=q['bytes'];ranges.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=q['bytes']))
    assert total<=650*1024*1024
    for q in manifest['files']:
        p=ROOT/q['local'];assert p.stat().st_size==q['bytes'] and sha(p)==q['sha256'] and q['zip_crc_checked']
    out=dict(complete_members=10,successful_cached_network_payload_bytes=total,payload_budget_bytes=650*1024*1024,final_attempt_new_network_bytes=manifest['new_network_bytes'],attempt1_terminal_error='RemoteDisconnected while opening final PNG ZIP header; session71236 exit1',attempt2='Same frozen script verified and skipped nine complete files, then retrieved missing PNG.',scope='Byte total is successful cached Range payloads across both attempts; excludes protocol overhead/no-response connection.',range_receipts=ranges,selection_sha256=sha(OUT/'selection.json'),manifest_sha256=sha(OUT/'manifest.json'),attempts_sha256=sha(OUT/'acquisition_attempts.json'),script_sha256=sha(Path(__file__)),training=False)
    target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='range_receipts'}),flush=True)
if __name__=='__main__':main()
