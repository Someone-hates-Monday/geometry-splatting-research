"""Archive selected public TGS sources for static reading, never execute them.

This is source retrieval, not a renderer/experiment/test runner. Git blob hashes
from the pinned recursive trees are checked before each immutable file is saved.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent / 'sources' / 'tgs_subdivision_static_v1'
PINS = {
    'triangle-splatting': ('685ff270a8273afc3a2b094ae3148dd0b98e31e9', [
        '.gitmodules', 'LICENSE.md', 'LICENSE_GS.md', 'README.md',
        'scene/triangle_model.py', 'triangle_renderer/__init__.py',
        'arguments/__init__.py', 'train.py', 'utils/loss_utils.py',
    ]),
    'diff-triangle-rasterization': ('6d61f4c0a571c635e37ffb740250706fa30a5541', [
        '.gitmodules', 'LICENSE.md', 'LICENSE_GS.md', 'README.md',
        'cuda_rasterizer/forward.cu', 'cuda_rasterizer/backward.cu',
        'cuda_rasterizer/auxiliary.h', 'cuda_rasterizer/rasterizer_impl.cu',
        'cuda_rasterizer/config.h', 'diff_triangle_rasterization/__init__.py',
    ]),
}


def get(url):
    with urlopen(Request(url, headers={'User-Agent': 'GGGS-research-source-audit'}), timeout=30) as response:
        return response.read()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f'Refusing to overwrite differing source: {path}')
    else:
        path.write_bytes(data)


def main():
    if (ROOT / 'manifest.json').exists():
        raise RuntimeError('Archive already complete; read its manifest instead of re-running.')
    records, repos = [], []
    for repo, (commit, paths) in PINS.items():
        api = f'https://api.github.com/repos/trianglesplatting/{repo}'
        commit_path = ROOT / repo / 'metadata' / 'commit.json'
        commit_bytes = commit_path.read_bytes() if commit_path.exists() else get(f'{api}/commits/{commit}')
        commit_record = json.loads(commit_bytes)
        tree_sha = commit_record['commit']['tree']['sha']
        tree_url = f'{api}/git/trees/{tree_sha}?recursive=1'
        tree_path = ROOT / repo / 'metadata' / 'tree.json'
        tree_bytes = tree_path.read_bytes() if tree_path.exists() else get(tree_url)
        tree = json.loads(tree_bytes)
        if tree['truncated']:
            raise RuntimeError('Truncated Git tree')
        index = {entry['path']: entry for entry in tree['tree']}
        if commit_record['sha'] != commit or tree_sha != tree['sha']:
            raise RuntimeError('Commit/tree identity mismatch')
        save(ROOT / repo / 'metadata' / 'tree.json', tree_bytes)
        save(ROOT / repo / 'metadata' / 'commit.json', commit_bytes)
        if repo == 'triangle-splatting':
            gitlink = index['submodules/diff-triangle-rasterization']
            if gitlink['type'] != 'commit' or gitlink['sha'] != PINS['diff-triangle-rasterization'][0]:
                raise RuntimeError('Native submodule pin mismatch')
        repos.append({'repo': repo, 'commit': commit, 'tree_sha': tree['sha'],
                      'commit_date': commit_record['commit']['committer']['date'],
                      'gitlinks': [e for e in tree['tree'] if e['type'] == 'commit']})
        for relative in paths:
            entry = index[relative]
            if entry['type'] != 'blob':
                raise RuntimeError('Expected regular blob')
            url = f'https://raw.githubusercontent.com/trianglesplatting/{repo}/{commit}/{relative}'
            target = ROOT / repo / relative
            data = target.read_bytes() if target.exists() else get(url)
            blob = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
            if blob != entry['sha'] or len(data) != entry['size']:
                raise RuntimeError(f'Git blob mismatch: {repo}/{relative}')
            save(target, data)
            records.append({'repo': repo, 'path': relative, 'url': url,
                            'bytes': len(data), 'git_blob_sha1': blob,
                            'sha256': hashlib.sha256(data).hexdigest(),
                            'lines': len(data.splitlines())})
            print(f'archived {repo}/{relative}', flush=True)
    manifest = {'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
                'scope': 'selected static sources; not a complete build checkout',
                'downloaded_sources_executed': False, 'cuda_compilation': False,
                'experiments_started': False, 'repos': repos, 'files': records}
    save(ROOT / 'manifest.json', (json.dumps(manifest, indent=2) + '\n').encode())
    print(json.dumps({'status': 'complete', 'files': len(records),
                      'bytes': sum(r['bytes'] for r in records), 'manifest': str(ROOT / 'manifest.json')}))


if __name__ == '__main__':
    main()
