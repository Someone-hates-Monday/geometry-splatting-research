"""Wait for the initial uploader, then resume and verify; durable local supervisor."""
import ctypes,json,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'.asset-transfer'
REPO='Someone-hates-Monday/geometry-splatting-research';TAG='research-assets-2026-11-15'
def status():
    receipts=[json.loads(p.read_text(encoding='utf8')) for p in OUT.glob('*.receipt.json')]
    x=dict(updated_utc=datetime.now(timezone.utc).isoformat(),verified_volumes=len(receipts),verified_compressed_bytes=sum(r['bytes'] for r in receipts),planned_volumes=90,complete=(OUT/'COMPLETE.json').exists())
    p=OUT/'TRANSFER_STATUS.json';p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf8')
    subprocess.run(['gh','release','upload',TAG,str(p),'--repo',REPO,'--clobber'],timeout=90,check=False)
    print(json.dumps(x),flush=True)
def main():
    k=ctypes.windll.kernel32;k.OpenProcess.restype=ctypes.c_void_p
    h=k.OpenProcess(0x100000,False,int(sys.argv[1]))
    if h:
        try:
            while k.WaitForSingleObject(ctypes.c_void_p(h),30000)==258:
                try:status()
                except Exception as e:print(str(e),flush=True)
        finally:k.CloseHandle(ctypes.c_void_p(h))
    for attempt in range(4):
        r=subprocess.Popen([sys.executable,str(ROOT/'handoff/upload_research_assets.py')],cwd=ROOT)
        while r.poll() is None:
            time.sleep(30)
            try:status()
            except Exception as e:print(str(e),flush=True)
        if r.returncode==0 and (OUT/'COMPLETE.json').exists():status();return
        time.sleep(30)
    raise RuntimeError('Transfer retries exhausted; inspect local logs and resume uploader.')
if __name__=='__main__':main()
