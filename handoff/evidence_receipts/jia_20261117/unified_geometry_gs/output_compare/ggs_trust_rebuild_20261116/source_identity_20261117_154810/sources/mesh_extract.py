import json
import os
from argparse import ArgumentParser

import numpy as np
import open3d as o3d
import open3d.core as o3c
import torch

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.general_utils import safe_state
from utils.jia_io_guard import DEFAULT_MESH_ARTIFACTS, assert_fresh_output_dir
from utils.loss_utils import PatchMatch


def post_process_mesh(mesh, cluster_to_keep=1):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy

    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = mesh_0.cluster_connected_triangles()

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50)  # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0


def _apply_gt_mask(depth_t: torch.Tensor, viewpoint_cam) -> torch.Tensor:
    depth = depth_t.clone()
    if viewpoint_cam.gt_mask is not None:
        depth[(viewpoint_cam.gt_mask < 0.5)] = 0
    return depth


def _trust_mask_depth(
    *,
    gaussians,
    pipe,
    kernel_size,
    patchmatch: PatchMatch,
    viewpoint_cam,
    train_cams,
    background,
    ncc_mask_thresh: float,
):
    """Render once; build keep mask with original render_pkg; zero depth copy outside K."""
    render_pkg = render(viewpoint_cam, gaussians, pipe, background, kernel_size)
    depth_raw = render_pkg["median_depth"].clone()
    depth_gt = _apply_gt_mask(depth_raw, viewpoint_cam)

    # B: finite positive depth after gt_mask
    depth_hw = depth_gt.squeeze(0)
    B = torch.isfinite(depth_hw) & (depth_hw > 0)

    nearest_cam = None
    nearest_name = None
    if getattr(viewpoint_cam, "nearest_id", None) and len(viewpoint_cam.nearest_id) > 0:
        nearest_cam = train_cams[viewpoint_cam.nearest_id[0]]
        nearest_name = getattr(nearest_cam, "image_name", str(viewpoint_cam.nearest_id[0]))

    pm_out = patchmatch(
        gaussians,
        render_pkg,  # original pkg for warp — do not zero depths first
        viewpoint_cam,
        nearest_cam,
        iteration=0,
        depth_normal=None,
        return_trust_maps=True,
        ncc_mask_thresh=float(ncc_mask_thresh),
    )
    trust_maps = pm_out[-1]
    keep = trust_maps["keep"]
    # Final K = B & d_mask & evaluated & warp_valid & isfinite(error) & (error<tau)
    # trust keep already encodes geometry/eval/warp/error<tau; still require isfinite(error).
    err = trust_maps["error"]
    evaluated = trust_maps["evaluated"]
    geometry_valid = trust_maps["geometry_valid"]
    warp_valid = trust_maps["warp_valid"]
    K = B & keep & torch.isfinite(err)

    depth_out = torch.where(K, depth_hw, torch.zeros_like(depth_hw))
    # non-finite -> 0
    depth_out = torch.where(torch.isfinite(depth_out), depth_out, torch.zeros_like(depth_out))

    # Mutual exclusive stats over B (keep three-way totals + subdivided other_fail)
    n_B = int(B.sum().item())
    n_keep = int(K.sum().item())
    ncc_reject = B & geometry_valid & evaluated & warp_valid & torch.isfinite(err) & (err >= float(ncc_mask_thresh))
    other_fail = B & (~K) & (~ncc_reject)
    geom_fail = B & (~geometry_valid)
    not_evaluated = B & geometry_valid & (~evaluated)
    warp_fail = B & geometry_valid & evaluated & (~warp_valid)
    nonfinite_err = B & geometry_valid & evaluated & warp_valid & (~torch.isfinite(err))
    stats = {
        "image_name": viewpoint_cam.image_name,
        "nearest_image_name": nearest_name,
        "nearest_id0": int(viewpoint_cam.nearest_id[0]) if nearest_cam is not None else None,
        "H": int(depth_hw.shape[0]),
        "W": int(depth_hw.shape[1]),
        "n_B": n_B,
        "n_keep": n_keep,
        "n_ncc_reject": int(ncc_reject.sum().item()),
        "n_geom_or_warp_fail": int(other_fail.sum().item()),
        "n_geom_fail": int(geom_fail.sum().item()),
        "n_not_evaluated": int(not_evaluated.sum().item()),
        "n_warp_fail": int(warp_fail.sum().item()),
        "n_nonfinite_error": int(nonfinite_err.sum().item()),
        "n_outside_B": int((~B).sum().item()),
        "n_nonfinite_depth": int((~torch.isfinite(depth_hw)).sum().item()),
        "frac_keep": (n_keep / n_B) if n_B > 0 else None,
        "frac_ncc_reject": (int(ncc_reject.sum().item()) / n_B) if n_B > 0 else None,
        "frac_geom_or_warp_fail": (int(other_fail.sum().item()) / n_B) if n_B > 0 else None,
        "counts_sum_eq_B": (n_keep + int(ncc_reject.sum().item()) + int(other_fail.sum().item())) == n_B,
        "other_fail_subdiv_sum_eq": (
            int(geom_fail.sum().item())
            + int(not_evaluated.sum().item())
            + int(warp_fail.sum().item())
            + int(nonfinite_err.sum().item())
        )
        == int(other_fail.sum().item()),
        "depth_nonzero_eq_keep": int((depth_out > 0).sum().item()) == n_keep,
    }
    rendered_img = torch.clamp(render_pkg["render"], min=0, max=1.0).cpu().numpy().transpose(1, 2, 0)
    return (
        np.ascontiguousarray(rendered_img),
        depth_out.detach().cpu().numpy().astype(np.float32),
        stats,
        {k: v.detach().cpu() for k, v in trust_maps.items()},
        depth_hw.detach().cpu(),
        K.detach().cpu(),
    )


def extract_mesh(
    dataset,
    pipe,
    iteration,
    num_cluster=1,
    voxel_size=0.002,
    depth_scale=1.0,
    depth_max=8.0,
    trunc_voxel_multiplier=4.0,
    block_count=50000,
    write_dir=None,
    trust_fuse: bool = False,
    ncc_mask_thresh: float = 0.9,
    metrics_json: str | None = None,
):
    gaussians = GaussianModel(dataset.sh_degree, dataset.sg_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    kernel_size = dataset.kernel_size

    bg_color = [1, 1, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    viewpoint_cam_list = scene.getTrainCameras()

    depth_list = []
    color_list = []
    per_view = []

    if trust_fuse:
        patchmatch = PatchMatch(
            patch_size=3,
            pixel_noise_th=1.0,
            kernel_size=kernel_size,
            pipe=pipe,
            debug=False,
            model_path=dataset.model_path,
        )
        for viewpoint_cam in viewpoint_cam_list:
            color, depth, stats, _maps, _depth_hw, _K = _trust_mask_depth(
                gaussians=gaussians,
                pipe=pipe,
                kernel_size=kernel_size,
                patchmatch=patchmatch,
                viewpoint_cam=viewpoint_cam,
                train_cams=viewpoint_cam_list,
                background=background,
                ncc_mask_thresh=ncc_mask_thresh,
            )
            color_list.append(color)
            depth_list.append(np.ascontiguousarray(depth))
            per_view.append(stats)
            assert stats["counts_sum_eq_B"], stats
            assert stats["depth_nonzero_eq_keep"], stats
    else:
        for viewpoint_cam in viewpoint_cam_list:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, kernel_size)
            rendered_img = torch.clamp(render_pkg["render"], min=0, max=1.0).cpu().numpy().transpose(1, 2, 0)
            color_list.append(np.ascontiguousarray(rendered_img))
            depth = _apply_gt_mask(render_pkg["median_depth"], viewpoint_cam)
            depth_list.append(depth[0].cpu().numpy())

    torch.cuda.empty_cache()
    explicit_new_path = bool(write_dir)
    out_dir = write_dir if write_dir else dataset.model_path
    if explicit_new_path:
        out_dir = assert_fresh_output_dir(
            out_dir,
            artifacts=DEFAULT_MESH_ARTIFACTS,
            forbidden_roots=[dataset.model_path],
            label="mesh_extract",
        )
    else:
        os.makedirs(out_dir, exist_ok=True)
    print(
        f"[mesh_extract] out={out_dir} voxel={voxel_size} depth_scale={depth_scale} "
        f"depth_max={depth_max} trunc_voxel_multiplier={trunc_voxel_multiplier} "
        f"clusters={num_cluster} block_count={block_count} trust_fuse={trust_fuse} "
        f"ncc_mask_thresh={ncc_mask_thresh}"
    )

    if metrics_json or trust_fuse:
        metrics_path = metrics_json or os.path.join(out_dir, "per_view_metrics.json")
        if explicit_new_path and os.path.exists(metrics_path):
            raise RuntimeError(f"mesh_extract: refuse overwrite {metrics_path}")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({"trust_fuse": trust_fuse, "ncc_mask_thresh": ncc_mask_thresh, "views": per_view}, f, indent=2)
        print(f"[mesh_extract] wrote {metrics_path}")

    # Finer voxels need more blocks for the same scene extent.
    if voxel_size < 0.002 and block_count <= 50000:
        block_count = 120000

    # Fail closed if nothing to fuse
    n_pos = sum(int((d > 0).sum()) for d in depth_list)
    if n_pos == 0:
        raise RuntimeError("FAILED: no positive depth pixels to integrate (empty fusion)")

    o3d_device = o3d.core.Device("CPU:0")
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1), (1), (3)),
        voxel_size=float(voxel_size),
        block_resolution=16,
        block_count=int(block_count),
        device=o3d_device,
    )
    for color, depth, viewpoint_cam in zip(color_list, depth_list, viewpoint_cam_list):
        depth = o3d.t.geometry.Image(np.ascontiguousarray(depth))
        depth = depth.to(o3d_device)
        color = o3d.t.geometry.Image(color)
        color = color.to(o3d_device)
        intrinsic = np.array(
            [[viewpoint_cam.Fx, 0, viewpoint_cam.Cx], [0, viewpoint_cam.Fy, viewpoint_cam.Cy], [0, 0, 1]],
            dtype=np.float64,
        )
        intrinsic = o3d.core.Tensor(intrinsic)
        extrinsic = o3d.core.Tensor((viewpoint_cam.world_view_transform.T).cpu().numpy().astype(np.float64))
        frustum_block_coords = vbg.compute_unique_block_coordinates(
            depth, intrinsic, extrinsic, float(depth_scale), float(depth_max), float(trunc_voxel_multiplier)
        )
        vbg.integrate(
            frustum_block_coords,
            depth,
            color,
            intrinsic,
            extrinsic,
            float(depth_scale),
            float(depth_max),
            float(trunc_voxel_multiplier),
        )

    mesh = vbg.extract_triangle_mesh()
    mesh_legacy = mesh.to_legacy()
    if len(mesh_legacy.vertices) == 0 or len(mesh_legacy.triangles) == 0:
        raise RuntimeError("FAILED: empty mesh before post_process")
    mesh_legacy.compute_vertex_normals()
    recon_path = os.path.join(out_dir, "recon.ply")
    if explicit_new_path and os.path.exists(recon_path):
        raise RuntimeError(f"mesh_extract: refuse overwrite {recon_path}")
    if not o3d.io.write_triangle_mesh(recon_path, mesh_legacy):
        raise RuntimeError(f"FAILED: write_triangle_mesh returned false for {recon_path}")
    mesh_post = post_process_mesh(mesh_legacy, num_cluster)
    if len(mesh_post.vertices) == 0 or len(mesh_post.triangles) == 0:
        raise RuntimeError("FAILED: empty mesh after extract/post_process")
    recon_post_path = os.path.join(out_dir, "recon_post.ply")
    if explicit_new_path and os.path.exists(recon_post_path):
        raise RuntimeError(f"mesh_extract: refuse overwrite {recon_post_path}")
    if not o3d.io.write_triangle_mesh(recon_post_path, mesh_post):
        raise RuntimeError(f"FAILED: write_triangle_mesh returned false for {recon_post_path}")
    print("done!")


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--num_cluster", default=1, type=int)
    parser.add_argument("--voxel_size", type=float, default=0.002)
    parser.add_argument("--depth_scale", type=float, default=1.0)
    parser.add_argument("--depth_max", type=float, default=8.0)
    parser.add_argument(
        "--trunc_voxel_multiplier",
        type=float,
        default=4.0,
        help="TSDF truncation length in voxel units (Open3D trunc_voxel_multiplier). "
        "Default 4.0 from extract-sweep (ex_trunc4) on frozen no_dn; legacy was 8.0.",
    )
    parser.add_argument("--block_count", type=int, default=50000)
    parser.add_argument(
        "--write_dir",
        type=str,
        default="",
        help="If set, write recon*.ply here instead of -m (load still uses -m).",
    )
    parser.add_argument(
        "--trust_fuse",
        action="store_true",
        help="Zero depths outside trusted NCC band (nearest_id[0], tau=ncc_mask_thresh) before TSDF.",
    )
    parser.add_argument("--ncc_mask_thresh", type=float, default=0.9)
    parser.add_argument("--metrics_json", type=str, default="")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    with torch.no_grad():
        extract_mesh(
            model.extract(args),
            pipeline.extract(args),
            args.iteration,
            num_cluster=args.num_cluster,
            voxel_size=args.voxel_size,
            depth_scale=args.depth_scale,
            depth_max=args.depth_max,
            trunc_voxel_multiplier=args.trunc_voxel_multiplier,
            block_count=args.block_count,
            write_dir=(args.write_dir or None),
            trust_fuse=bool(args.trust_fuse),
            ncc_mask_thresh=float(args.ncc_mask_thresh),
            metrics_json=(args.metrics_json or None),
        )
