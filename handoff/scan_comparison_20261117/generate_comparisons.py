"""Regenerate the three same-view G-GS / Pipeline mesh comparisons.

Run in the D-drive Ubuntu WSL environment with Open3D, NumPy and Pillow:
    python generate_comparisons.py
Optionally pass --root /path/to/extracted/assets --outdir /path/to/output.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/opt/geometry-compare-assets')
METRIC = 'unified_geometry_gs/output_compare'
FONT_CANDIDATES = (
    '/mnt/c/Windows/Fonts/msyh.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
)
FOV_SCALE = 1.45
DELTAS = {83: -0.0848401088, 63: -0.0685502445, 55: 0.0035933051}


def render(mesh_path, camera, width):
    native_w, native_h = camera['width'], camera['height']
    height = round(width * native_h / native_w)
    sx, sy = width / native_w, height / native_h
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, camera['fx'] * sx / FOV_SCALE, camera['fy'] * sy / FOV_SCALE, width / 2, height / 2)
    c2w = np.eye(4)
    c2w[:3, :3] = np.array(camera['rotation'])
    c2w[:3, 3] = camera['position']
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([1, 1, 1, 1])
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.compute_vertex_normals()
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = 'defaultLit'
    material.base_color = [1, 1, 1, 1]
    renderer.scene.add_geometry('mesh', mesh, material)
    renderer.setup_camera(intrinsic, np.linalg.inv(c2w))
    image = Image.fromarray(np.asarray(renderer.render_to_image()).copy())
    return image, {'width': width, 'height': height, 'fx': intrinsic.get_focal_length()[0], 'fy': intrinsic.get_focal_length()[1], 'cx': width / 2, 'cy': height / 2, 'extrinsic_world_to_camera': np.linalg.inv(c2w).tolist(), 'triangles': len(mesh.triangles), 'vertices': len(mesh.vertices)}


def render_one(scan, out, width):
    a = ROOT / METRIC / 'ggs_baseline' / f'scan{scan}'
    b = ROOT / METRIC / 'ggs_pipeline' / f'scan{scan}'
    cams_a, cams_b = [json.loads((x / 'cameras.json').read_text()) for x in (a, b)]
    ca = next(x for x in cams_a if x['img_name'] == '0000')
    cb = next(x for x in cams_b if x['img_name'] == '0000')
    if ca != cb:
        raise RuntimeError('baseline/pipeline camera mismatch')
    paths = [a / 'recon_post.ply', b / 'recon_post.ply']
    images, settings = zip(*(render(p, ca, width) for p in paths))
    if (settings[0]['width'], settings[0]['height']) != (settings[1]['width'], settings[1]['height']):
        raise RuntimeError('render dimensions mismatch')
    w, h = images[0].size
    title_h, footer_h = 92, 56
    contact = Image.new('RGB', (2 * w, h + title_h + footer_h), '#ffffff')
    draw = ImageDraw.Draw(contact)
    font_path = next((p for p in FONT_CANDIDATES if Path(p).is_file()), FONT_CANDIDATES[-1])
    font = ImageFont.truetype(font_path, 28)
    small = ImageFont.truetype(font_path, 21)
    for index, (image, title) in enumerate(zip(images, ('原始 G-GS（Baseline）', 'Pipeline（既有优化配方）'))):
        contact.paste(image, (index * w, title_h))
        draw.text((index * w + 24, 12), title, fill='#111827', font=font)
    delta = DELTAS[scan]
    result = '退化案例' if delta > 0 else '改善'
    draw.text((24, 54), f'scan{scan}  |  相机 0000  |  历史 overall Δ = {delta:+.10f} mm（{result}）', fill='#374151', font=small)
    draw.line((w, title_h, w, h + title_h), fill='#cbd5e1', width=2)
    draw.line((0, h + title_h, 2 * w, h + title_h), fill='#cbd5e1', width=1)
    draw.text((24, h + title_h + 13), '相同相机与渲染设置；视野扩大 1.45 倍，以展示完整物体。数值来自历史 DTU 网格评测。', fill='#4b5563', font=small)
    out.parent.mkdir(parents=True, exist_ok=True)
    contact.save(out)
    print(json.dumps({'scan': scan, 'out': str(out), 'settings': settings[0], 'fov_scale': FOV_SCALE, 'mesh_sha256': [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]}))


def main():
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT, help='Extracted research assets root')
    parser.add_argument('--outdir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--scans', type=int, nargs='+', choices=sorted(DELTAS), default=[83, 63, 55])
    parser.add_argument('--width', type=int, default=1000, help='Width of each panel')
    args = parser.parse_args()
    ROOT = args.root
    for scan in args.scans:
        render_one(scan, args.outdir / f'scan{scan}_same_view.png', args.width)


if __name__ == '__main__':
    main()
