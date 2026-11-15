"""Small offline tests; no credentials, real datasets, CUDA or network."""
import hashlib,io,tempfile,unittest,zipfile
from pathlib import Path
from deploy_assets import restore,digest,target

class RestoreTests(unittest.TestCase):
    def fixture(self,base):
        cache=base/'cache';cache.mkdir();entries=[];assets=[]
        for i,(name,data,extra) in enumerate([
            ('models/final.pth',b'checkpoint',{}),
            ('snapshots/model.zip.asset-part-000000000000',b'abc',dict(path='snapshots/model.zip',offset=0,length=3,bytes=6)),
            ('snapshots/model.zip.asset-part-000000000003',b'def',dict(path='snapshots/model.zip',offset=3,length=3,bytes=6))]):
            p=cache/f'volume{i}.zip'
            with zipfile.ZipFile(p,'w') as z:z.writestr(name,data)
            entries.append(dict(path=name,archive_path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())|extra)
            assets.append(dict(name=p.name,bytes=p.stat().st_size,sha256=digest(p)))
        return dict(entries=entries,assets=assets,skipped=[]),cache,base/'restored'
    def test_roundtrip_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            m,c,r=self.fixture(Path(d));x=restore(m,c,r)
            self.assertEqual((r/'models/final.pth').read_bytes(),b'checkpoint')
            self.assertEqual((r/'snapshots/model.zip').read_bytes(),b'abcdef')
            self.assertEqual(x['volumes'],3);restore(m,c,r)
    def test_corrupt_archive_stops_before_writes(self):
        with tempfile.TemporaryDirectory() as d:
            m,c,r=self.fixture(Path(d));(c/'volume2.zip').write_bytes(b'broken')
            with self.assertRaises(ValueError):restore(m,c,r)
            self.assertFalse((r/'models/final.pth').exists())
    def test_wrong_member_hash(self):
        with tempfile.TemporaryDirectory() as d:
            m,c,r=self.fixture(Path(d));m['entries'][0]['sha256']='0'*64
            with self.assertRaises(ValueError):restore(m,c,r)
            self.assertFalse((r/'models/final.pth').exists())
    def test_existing_different_file_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            m,c,r=self.fixture(Path(d));(r/'models').mkdir(parents=True);p=r/'models/final.pth';p.write_bytes(b'keep')
            with self.assertRaises(ValueError):restore(m,c,r)
            self.assertEqual(p.read_bytes(),b'keep')
    def test_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in ['../outside','/outside','C:/outside','foo\\bar','foo/../bar']:
                with self.assertRaises(ValueError):target(root,name)
    def test_noncontiguous_parts(self):
        with tempfile.TemporaryDirectory() as d:
            m,c,r=self.fixture(Path(d));m['entries'][2]['offset']=4
            with self.assertRaises(ValueError):restore(m,c,r)
            self.assertFalse((r/'snapshots/model.zip').exists())
    def test_symlink_destination_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'real').mkdir()
            try:(root/'link').symlink_to(root/'real',target_is_directory=True)
            except OSError:self.skipTest('Symlink permission unavailable')
            with self.assertRaises(ValueError):target(root,'link/model.pth')
if __name__=='__main__':unittest.main()
