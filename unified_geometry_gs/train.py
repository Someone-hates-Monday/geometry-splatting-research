#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import signal
import sys
import uuid
from argparse import ArgumentParser, Namespace
from random import randint, sample
from typing import Any, Sequence, TypedDict, Union

import cv2
import numpy as np
import torch
from tqdm import tqdm


def stop_request_path(model_path: str) -> str:
    return os.path.join(model_path, "STOP_AT_CHECKPOINT")


class SoftStopController:
    """Stop after the next scheduled checkpoint (or an emergency save if none remain).

    Arm via:
      - touch <model_path>/STOP_AT_CHECKPOINT
      - SIGUSR1
      - first Ctrl+C / SIGINT (second forces immediate KeyboardInterrupt)
    """

    def __init__(self, model_path: str, checkpoint_iterations: Sequence[int]):
        self.model_path = model_path
        self.ckpt_iters = sorted({int(x) for x in checkpoint_iterations})
        self.requested = False
        self.stop_at: int | None = None
        self._sigint_count = 0
        # Ignore a leftover stop file from a previous run.
        flag = stop_request_path(model_path)
        if os.path.isfile(flag):
            os.remove(flag)
            print(f"[soft-stop] cleared stale stop file: {flag}")
        self._install_handlers()

    def _install_handlers(self) -> None:
        def on_usr1(signum, frame):
            self.arm(reason=f"signal {signum}")

        def on_int(signum, frame):
            self._sigint_count += 1
            if self._sigint_count == 1:
                self.arm(reason="SIGINT/Ctrl+C (press again to abort immediately)")
            else:
                raise KeyboardInterrupt

        try:
            signal.signal(signal.SIGUSR1, on_usr1)
        except (ValueError, OSError, AttributeError):
            pass
        try:
            signal.signal(signal.SIGINT, on_int)
        except (ValueError, OSError, AttributeError):
            pass

    def arm(self, iteration: int | None = None, reason: str = "") -> None:
        if self.requested:
            return
        self.requested = True
        flag = stop_request_path(self.model_path)
        try:
            with open(flag, "w", encoding="utf-8") as f:
                f.write(reason or "stop requested\n")
        except OSError:
            pass
        if iteration is not None:
            self._set_stop_at(iteration)
        msg = f"[soft-stop] armed ({reason or 'request'})"
        if self.stop_at is not None:
            msg += f" — will save checkpoint at iter {self.stop_at} then exit"
        else:
            msg += " — stop_at will be set on next poll"
        print("\n" + msg, flush=True)

    def _set_stop_at(self, iteration: int) -> None:
        future = [c for c in self.ckpt_iters if c >= iteration]
        self.stop_at = future[0] if future else iteration

    def poll(self, iteration: int) -> None:
        if not self.requested and os.path.isfile(stop_request_path(self.model_path)):
            self.arm(iteration=iteration, reason="STOP_AT_CHECKPOINT file")
        elif self.requested and self.stop_at is None:
            self._set_stop_at(iteration)

    def should_save_checkpoint(self, iteration: int) -> bool:
        if iteration in self.ckpt_iters:
            return True
        if not self.requested:
            return False
        if self.stop_at is None:
            self._set_stop_at(iteration)
        # Emergency save when stopping at an unscheduled iteration.
        return self.stop_at == iteration

    def should_exit(self, iteration: int) -> bool:
        if not self.requested:
            return False
        if self.stop_at is None:
            self._set_stop_at(iteration)
        return iteration >= self.stop_at

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import network_gui, render
from losses import (
    compute_depth_occ_loss,
    compute_field_occ_loss,
    compute_normal_guided_ambiguity_loss,
    compute_occ_loss,
    compute_occlusion_ranking_loss,
    compute_thin_occ_boundary_loss,
    compute_w_geo_from_patchmatch,
    depth_normal_loss_weighted,
    edge_weights_from_depth,
)
from scene import GaussianModel, Scene
from scene.cameras import Camera
from utils.general_utils import safe_state
from utils.graphics_utils import depth_to_normal
from utils.image_utils import psnr
from utils.loss_utils import L1_loss_appearance, PatchMatch, l1_loss, ssim

from losses.m1_geometry_router import (
    M1GeometrySoftRouter,
    compute_m1_router_features,
    router_balance_loss,
)


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, dataset.sg_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint, map_location="cuda", weights_only=False)
        gaussians.restore(model_params, opt)
    soft_stop = SoftStopController(dataset.model_path, checkpoint_iterations)
    if soft_stop.ckpt_iters:
        print(f"[soft-stop] checkpoints at: {soft_stop.ckpt_iters}")
        print(f"[soft-stop] request stop: touch {stop_request_path(dataset.model_path)}")
        print("[soft-stop] or: kill -USR1 <train_pid>  |  Ctrl+C once (twice = abort now)")
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    kernel_size = dataset.kernel_size

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    trainCameras = scene.getTrainCameras().copy()
    if dataset.disable_filter3D:
        gaussians.reset_3D_filter()
    else:
        gaussians.compute_3D_filter(cameras=trainCameras)

    if opt.lambda_multi_view_ncc > 0 or opt.lambda_multi_view_geo > 0:
        patchmatch = PatchMatch(
            opt.multi_view_patch_size,
            opt.multi_view_pixel_noise_th,
            kernel_size=kernel_size,
            pipe=pipe,
            debug=True,
            model_path=dataset.model_path,
        )

    m1_router: M1GeometrySoftRouter | None = None
    m1_router_optimizer: torch.optim.Optimizer | None = None
    if getattr(opt, "use_m1_geometry_router", False):
        m1_router = M1GeometrySoftRouter(hidden_dim=int(opt.m1_router_hidden)).cuda()
        m1_router_optimizer = torch.optim.Adam(m1_router.parameters(), lr=float(opt.m1_router_lr))
        print(
            "[M1 router] geometry-aware soft routing enabled "
            f"(grid={opt.m1_router_grid_h}x{opt.m1_router_grid_w}, balance={opt.m1_router_balance})"
        )

    if float(getattr(opt, "depth_normal_edge_boost", 0.0)) > 0.0 and opt.lambda_depth_normal > 0:
        print(
            "[edge depth-normal] Sobel-weighted normal loss — boost="
            f"{opt.depth_normal_edge_boost} quantiles="
            f"[{getattr(opt, 'depth_normal_edge_q_low', 0.05)}, {getattr(opt, 'depth_normal_edge_q_high', 0.95)}]"
        )

    if bool(getattr(opt, "use_ncc_gate_dn", False)) and opt.lambda_depth_normal > 0:
        print(
            "[ncc-gate-dn] depth-normal weighted by sg(1-c) from warp NCC correlation "
            f"(lambda_depth_normal={opt.lambda_depth_normal})"
        )

    if float(getattr(opt, "lambda_ncc_fail", 0.0)) > 0:
        print(
            "[ncc-fail] coverage on warp-ok / ncc-mask-fail pixels "
            f"(lambda_ncc_fail={opt.lambda_ncc_fail})"
        )

    _mv_ncc_views = max(1, int(getattr(opt, "multi_view_ncc_views", 1)))
    if _mv_ncc_views > 1 and (opt.lambda_multi_view_ncc > 0 or opt.lambda_multi_view_geo > 0):
        print(
            f"[mv-ncc] ordered top-{_mv_ncc_views} nearest views "
            "(average trusted-band NCC/geo)"
        )

    _ncc_tau = float(getattr(opt, "ncc_mask_thresh", 0.9))
    if abs(_ncc_tau - 0.9) > 1e-9 and (opt.lambda_multi_view_ncc > 0 or opt.lambda_multi_view_geo > 0):
        print(f"[ncc-tau] trusted-band threshold ncc_mask_thresh={_ncc_tau}")

    if float(getattr(opt, "multi_view_edge_boost", 0.0)) > 0.0 and (
        opt.lambda_multi_view_geo > 0 or opt.lambda_multi_view_ncc > 0
    ):
        geo_only = bool(getattr(opt, "multi_view_edge_geo_only", False))
        print(
            "[edge patchmatch] depth Sobel weights on "
            f"{'geo only' if geo_only else 'geo + NCC'} — boost="
            f"{opt.multi_view_edge_boost} quantiles="
            f"[{getattr(opt, 'multi_view_edge_q_low', 0.05)}, {getattr(opt, 'multi_view_edge_q_high', 0.95)}]"
        )
        if (
            not geo_only
            and opt.lambda_multi_view_ncc > 0
            and opt.lambda_multi_view_geo > 0
        ):
            print(
                "[edge patchmatch] hint: if mesh metrics regress vs baseline, try "
                "`--multi_view_edge_geo_only` and/or lower `--multi_view_edge_boost` "
                "(see scripts/train_pm_edge_round2.sh)."
            )

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_normal_loss_for_log = 0.0
    ema_ncc_loss_for_log = 0.0
    ema_ncc_fail_loss_for_log = 0.0
    ema_occ_loss_for_log = 0.0
    ema_depth_occ_loss_for_log = 0.0
    ema_depth_occ_rank_loss_for_log = 0.0
    ema_occ_thin_loss_for_log = 0.0
    ema_occ_ng_loss_for_log = 0.0
    ema_field_occ_loss_for_log = 0.0
    _field_occ_warned = False
    os.makedirs(os.path.join(dataset.model_path, "debug"), exist_ok=True)
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        soft_stop.poll(iteration)
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                (
                    custom_cam,
                    do_training,
                    pipe.convert_SHs_python,
                    pipe.compute_cov3D_python,
                    keep_alive,
                    scaling_modifer,
                ) = network_gui.receive()
                if custom_cam != None:
                    net_image = render(
                        custom_cam,
                        gaussians,
                        pipe,
                        background,
                        kernel_size,
                        scaling_modifer,
                    )["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            if gaussians.max_sh_degree == gaussians.max_sh_degree:
                gaussians.unlockSGdegree(100)
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam: Camera = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        reg_kick_on = iteration >= opt.regularization_from_iter
        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe,
            background,
            kernel_size,
            require_depth=reg_kick_on,
        )
        rendered_image: torch.Tensor
        rendered_image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )
        gt_image = viewpoint_cam.original_image.cuda()

        Ll1_render = L1_loss_appearance(rendered_image, gt_image, gaussians, viewpoint_cam.uid)
        # Depth-derived normals + per-pixel normal error (depth-normal loss and/or route-B occ term)
        depth_normal = None
        depth_normal_loss = torch.tensor([0], dtype=torch.float32, device="cuda")
        normal_error_map: torch.Tensor | None = None
        normal_valid_mask: torch.Tensor | None = None
        use_ncc_gate_dn = bool(getattr(opt, "use_ncc_gate_dn", False))
        need_depth_geom = reg_kick_on and (
            opt.lambda_depth_normal > 0
            or opt.lambda_occ_ng_sharp > 0
            or opt.lambda_multi_view_ncc > 0
            or opt.lambda_multi_view_geo > 0
            or use_ncc_gate_dn
        )
        if need_depth_geom:
            depth_map: torch.Tensor = render_pkg["median_depth"]
            rendered_normal: torch.Tensor = render_pkg["normal"]
            depth_normal, valid_points = depth_to_normal(viewpoint_cam, depth_map)
            normal_error_map = 1 - torch.linalg.vecdot(rendered_normal, depth_normal, dim=0)
            normal_valid_mask = valid_points.squeeze()
            # Uniform / edge DN now; NCC-gated DN is applied after PatchMatch (needs corr map).
            if opt.lambda_depth_normal > 0 and not use_ncc_gate_dn:
                edge_boost = float(getattr(opt, "depth_normal_edge_boost", 0.0))
                if edge_boost > 0.0:
                    q_lo = float(getattr(opt, "depth_normal_edge_q_low", 0.05))
                    q_hi = float(getattr(opt, "depth_normal_edge_q_high", 0.95))
                    w_map = edge_weights_from_depth(
                        depth_map,
                        normal_valid_mask,
                        boost=edge_boost,
                        q_low=q_lo,
                        q_high=q_hi,
                    )
                    depth_normal_loss = depth_normal_loss_weighted(
                        normal_error_map,
                        normal_valid_mask,
                        w_map,
                    )
                else:
                    depth_normal_loss = torch.where(
                        normal_valid_mask, normal_error_map, torch.zeros_like(normal_error_map)
                    ).mean()

        # patch match loss (+ optional M2 / fail-cover / corr-map / multi-nearest extras)
        ncc_corr_map: torch.Tensor | None = None
        ncc_fail_loss = torch.tensor([0], dtype=torch.float32, device="cuda")
        lambda_ncc_fail = float(getattr(opt, "lambda_ncc_fail", 0.0))
        want_fail_cover = lambda_ncc_fail > 0
        want_corr_map = use_ncc_gate_dn and opt.lambda_depth_normal > 0
        if reg_kick_on and (opt.lambda_multi_view_ncc > 0 or opt.lambda_multi_view_geo or want_fail_cover):
            n_mv = max(1, int(getattr(opt, "multi_view_ncc_views", 1)))
            train_cams = scene.getTrainCameras()
            if len(viewpoint_cam.nearest_id) == 0:
                nearest_cams = [None]
            else:
                # Ordered top-K (nearest_id already sorted by distance/angle).
                # Legacy n_mv=1 used random sample among nearest_id; keep that for parity.
                if n_mv == 1:
                    nearest_cams = [train_cams[sample(viewpoint_cam.nearest_id, 1)[0]]]
                else:
                    nearest_cams = [train_cams[i] for i in viewpoint_cam.nearest_id[:n_mv]]
            _pm_edge_kw = dict(
                edge_boost=float(getattr(opt, "multi_view_edge_boost", 0.0)),
                edge_q_low=float(getattr(opt, "multi_view_edge_q_low", 0.05)),
                edge_q_high=float(getattr(opt, "multi_view_edge_q_high", 0.95)),
                edge_geo_only=bool(getattr(opt, "multi_view_edge_geo_only", False)),
                ncc_corr_power=(
                    float(getattr(opt, "ncc_corr_power", 1.0))
                    if bool(getattr(opt, "use_ncc_corr_weight", False))
                    else 0.0
                ),
                ncc_mask_thresh=float(getattr(opt, "ncc_mask_thresh", 0.9)),
                ncc_exact_gradient=bool(getattr(opt, "ncc_exact_gradient", False)),
                return_corr_map=want_corr_map,
                return_fail_cover=want_fail_cover,
            )

            def _unpack_pm(pm_out, *, with_stats: bool):
                """ncc, geo, [valid_ratio], [corr_map], [fail_cover] in that order."""
                vals = list(pm_out)
                ncc_l = vals.pop(0)
                geo_l = vals.pop(0)
                v_ratio = vals.pop(0) if with_stats else None
                corr = vals.pop(0) if want_corr_map else None
                fail_l = vals.pop(0) if want_fail_cover else None
                return ncc_l, geo_l, v_ratio, corr, fail_l

            ncc_acc: list[torch.Tensor] = []
            geo_acc: list[torch.Tensor] = []
            fail_acc: list[torch.Tensor] = []
            corr_acc: list[torch.Tensor] = []
            vratio_acc: list[torch.Tensor] = []

            for nearest_cam in nearest_cams:
                if opt.use_w_geo:
                    pm_out = patchmatch(
                        gaussians,
                        render_pkg,
                        viewpoint_cam,
                        nearest_cam,
                        iteration,
                        depth_normal,
                        return_stats=True,
                        **_pm_edge_kw,
                    )
                    ncc_l, geo_l, v_ratio, corr, fail_l = _unpack_pm(pm_out, with_stats=True)
                    if v_ratio is not None:
                        vratio_acc.append(v_ratio)
                else:
                    pm_out = patchmatch(
                        gaussians,
                        render_pkg,
                        viewpoint_cam,
                        nearest_cam,
                        iteration,
                        depth_normal,
                        **_pm_edge_kw,
                    )
                    ncc_l, geo_l, _, corr, fail_l = _unpack_pm(pm_out, with_stats=False)
                ncc_acc.append(ncc_l)
                geo_acc.append(geo_l)
                if fail_l is not None:
                    fail_acc.append(fail_l)
                if corr is not None:
                    corr_acc.append(corr)

            ncc_loss = torch.stack([t.reshape(()) for t in ncc_acc]).mean()
            geo_loss = torch.stack([t.reshape(()) for t in geo_acc]).mean()
            if fail_acc:
                ncc_fail_loss = torch.stack([t.reshape(()) for t in fail_acc]).mean()
            if corr_acc:
                ncc_corr_map = torch.stack(corr_acc, dim=0).mean(dim=0)
            if opt.use_w_geo:
                valid_ratio = (
                    torch.stack([t.reshape(()) for t in vratio_acc]).mean()
                    if vratio_acc
                    else torch.zeros((), dtype=torch.float32, device="cuda")
                )
                w_geo = compute_w_geo_from_patchmatch(
                    geo_loss=geo_loss,
                    valid_ratio=valid_ratio,
                    beta=float(opt.w_geo_beta),
                    min_w=float(opt.w_geo_min),
                )
                ncc_loss = w_geo * ncc_loss
                if not opt.w_geo_ncc_only:
                    geo_loss = w_geo * geo_loss
        else:
            ncc_loss = torch.tensor([0], dtype=torch.float32, device="cuda")
            geo_loss = torch.tensor([0], dtype=torch.float32, device="cuda")

        # NCC-complementary depth-normal: w = sg(1 - c); high NCC corr ⇒ suppress DN.
        if (
            reg_kick_on
            and use_ncc_gate_dn
            and opt.lambda_depth_normal > 0
            and normal_error_map is not None
            and normal_valid_mask is not None
        ):
            if ncc_corr_map is None:
                ncc_corr_map = torch.zeros_like(normal_error_map)
            w_dn = (1.0 - ncc_corr_map).clamp(0.0, 1.0)
            edge_boost = float(getattr(opt, "depth_normal_edge_boost", 0.0))
            if edge_boost > 0.0:
                depth_map = render_pkg["median_depth"]
                q_lo = float(getattr(opt, "depth_normal_edge_q_low", 0.05))
                q_hi = float(getattr(opt, "depth_normal_edge_q_high", 0.95))
                w_edge = edge_weights_from_depth(
                    depth_map,
                    normal_valid_mask,
                    boost=edge_boost,
                    q_low=q_lo,
                    q_high=q_hi,
                )
                w_dn = w_dn * w_edge
            depth_normal_loss = depth_normal_loss_weighted(
                normal_error_map,
                normal_valid_mask,
                w_dn,
            )

        # M1: unified occupancy-depth losses from same render outputs (alpha/depth)
        if reg_kick_on and (
            opt.lambda_occ > 0
            or opt.lambda_depth_occ > 0
            or opt.lambda_depth_occ_rank > 0
            or opt.lambda_occ_thin > 0
        ):
            alpha_map: torch.Tensor = render_pkg["mask"]
            median_depth_map: torch.Tensor = render_pkg["median_depth"]
            occ_loss = compute_occ_loss(
                alpha_map,
                entropy_weight=opt.occ_entropy_weight,
                sparsity_weight=opt.occ_sparsity_weight,
            )
            depth_occ_loss = compute_depth_occ_loss(
                median_depth_map,
                alpha_map,
                min_alpha=opt.depth_occ_min_alpha,
            )
            depth_occ_rank_loss = compute_occlusion_ranking_loss(
                median_depth_map,
                alpha_map,
                min_alpha=opt.depth_occ_min_alpha,
                margin=float(opt.occ_rank_margin),
                alpha_soft=float(opt.occ_rank_alpha_soft),
            )
            occ_thin_loss = compute_thin_occ_boundary_loss(
                median_depth_map,
                alpha_map,
                min_alpha=opt.depth_occ_min_alpha,
                edge_power=float(opt.occ_thin_edge_power),
            )
        else:
            occ_loss = torch.tensor([0], dtype=torch.float32, device="cuda")
            depth_occ_loss = torch.tensor([0], dtype=torch.float32, device="cuda")
            depth_occ_rank_loss = torch.tensor([0], dtype=torch.float32, device="cuda")
            occ_thin_loss = torch.tensor([0], dtype=torch.float32, device="cuda")

        # A: field-intrinsic occupancy on T=1/2 interface slope (no screen L_occ/Doc)
        field_occ_loss = torch.tensor([0], dtype=torch.float32, device="cuda")
        if reg_kick_on and float(getattr(opt, "lambda_field_occ", 0.0)) > 0:
            slope_map = render_pkg.get("interface_slope")
            if slope_map is None:
                raise RuntimeError(
                    "lambda_field_occ>0 but render_pkg has no interface_slope; "
                    "rebuild diff-gaussian-rasterization with 2-channel mdepth pack."
                )
            field_occ_loss = compute_field_occ_loss(
                slope_map,
                beta=float(getattr(opt, "field_occ_beta", 1.0)),
                clamp_to_median_mult=float(getattr(opt, "field_occ_clamp_mult", 10.0)),
            )
            if (
                not _field_occ_warned
                and float(slope_map.detach().max()) <= 0.0
            ):
                print(
                    "[A field-occ] warning: interface_slope is all zeros — "
                    "rebuild diff-gaussian-rasterization (2-channel mdepth) "
                    "or the field loss cannot train."
                )
                _field_occ_warned = True

        occ_ng_loss = torch.tensor([0], dtype=torch.float32, device="cuda")
        if (
            reg_kick_on
            and opt.lambda_occ_ng_sharp > 0
            and normal_error_map is not None
            and normal_valid_mask is not None
        ):
            occ_ng_loss = compute_normal_guided_ambiguity_loss(
                render_pkg["mask"],
                normal_error_map,
                normal_valid_mask,
                beta=float(opt.occ_ng_sharp_beta),
            )

        rgb_loss = (1.0 - opt.lambda_dssim) * Ll1_render + opt.lambda_dssim * (1.0 - ssim(rendered_image.unsqueeze(0), gt_image.unsqueeze(0)))

        m1_active = reg_kick_on and (
            opt.lambda_occ > 0
            or opt.lambda_depth_occ > 0
            or opt.lambda_depth_occ_rank > 0
            or opt.lambda_occ_thin > 0
        )
        use_router = (
            m1_active
            and m1_router is not None
            and getattr(opt, "use_m1_geometry_router", False)
        )
        _router_snap: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        if use_router:
            _alpha_r: torch.Tensor = render_pkg["mask"]
            _depth_r: torch.Tensor = render_pkg["median_depth"]
            feats = compute_m1_router_features(
                _depth_r,
                _alpha_r,
                int(opt.m1_router_grid_h),
                int(opt.m1_router_grid_w),
                float(opt.depth_occ_min_alpha),
            )
            if feats is None:
                w_occ = w_doc = w_rank = w_thin = torch.ones((), device="cuda", dtype=torch.float32)
                bal_loss = torch.zeros((), device="cuda", dtype=torch.float32)
            else:
                probs, mean_mult = m1_router(feats, float(opt.m1_router_temperature))
                w_occ, w_doc, w_rank, w_thin = mean_mult[0], mean_mult[1], mean_mult[2], mean_mult[3]
                bal_loss = router_balance_loss(probs)
                _router_snap = (
                    probs.detach().mean(dim=0),
                    mean_mult.detach(),
                    bal_loss.detach(),
                )
            m1_term = (
                opt.lambda_occ * w_occ * occ_loss
                + opt.lambda_depth_occ * w_doc * depth_occ_loss
                + opt.lambda_depth_occ_rank * w_rank * depth_occ_rank_loss
                + opt.lambda_occ_thin * w_thin * occ_thin_loss
            )
            m1_router_extra = float(opt.m1_router_balance) * bal_loss
        else:
            m1_term = (
                opt.lambda_occ * occ_loss
                + opt.lambda_depth_occ * depth_occ_loss
                + opt.lambda_depth_occ_rank * depth_occ_rank_loss
                + opt.lambda_occ_thin * occ_thin_loss
            )
            m1_router_extra = torch.zeros((), device="cuda", dtype=torch.float32)

        loss = (
            rgb_loss
            + opt.lambda_depth_normal * depth_normal_loss
            + opt.lambda_occ_ng_sharp * occ_ng_loss
            + opt.lambda_multi_view_ncc * ncc_loss
            + opt.lambda_multi_view_geo * geo_loss
            + lambda_ncc_fail * ncc_fail_loss
            + m1_term
            + m1_router_extra
            + float(getattr(opt, "lambda_field_occ", 0.0)) * field_occ_loss
        )

        if m1_router_optimizer is not None:
            m1_router_optimizer.zero_grad(set_to_none=True)
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_normal_loss_for_log = 0.4 * depth_normal_loss.item() + 0.6 * ema_normal_loss_for_log
            ema_ncc_loss_for_log = 0.4 * ncc_loss.item() + 0.6 * ema_ncc_loss_for_log
            ema_ncc_fail_loss_for_log = 0.4 * ncc_fail_loss.item() + 0.6 * ema_ncc_fail_loss_for_log
            ema_occ_loss_for_log = 0.4 * occ_loss.item() + 0.6 * ema_occ_loss_for_log
            ema_depth_occ_loss_for_log = 0.4 * depth_occ_loss.item() + 0.6 * ema_depth_occ_loss_for_log
            ema_depth_occ_rank_loss_for_log = (
                0.4 * depth_occ_rank_loss.item() + 0.6 * ema_depth_occ_rank_loss_for_log
            )
            ema_occ_thin_loss_for_log = 0.4 * occ_thin_loss.item() + 0.6 * ema_occ_thin_loss_for_log
            ema_occ_ng_loss_for_log = 0.4 * occ_ng_loss.item() + 0.6 * ema_occ_ng_loss_for_log
            ema_field_occ_loss_for_log = 0.4 * field_occ_loss.item() + 0.6 * ema_field_occ_loss_for_log

            if iteration % 10 == 0:
                postfix = {
                    "Loss": f"{ema_loss_for_log:.{4}f}",
                    "loss_normal": f"{ema_normal_loss_for_log:.{4}f}",
                    "loss_ncc": f"{ema_ncc_loss_for_log:.{4}f}",
                    "loss_occ": f"{ema_occ_loss_for_log:.{4}f}",
                    "loss_depth_occ": f"{ema_depth_occ_loss_for_log:.{4}f}",
                    "loss_occ_r": f"{ema_depth_occ_rank_loss_for_log:.{4}f}",
                    "loss_occ_thin": f"{ema_occ_thin_loss_for_log:.{4}f}",
                }
                if opt.lambda_occ_ng_sharp > 0:
                    postfix["loss_occ_ng"] = f"{ema_occ_ng_loss_for_log:.{4}f}"
                if float(getattr(opt, "lambda_field_occ", 0.0)) > 0:
                    postfix["loss_field_occ"] = f"{ema_field_occ_loss_for_log:.{4}f}"
                if float(getattr(opt, "lambda_ncc_fail", 0.0)) > 0:
                    postfix["loss_ncc_fail"] = f"{ema_ncc_fail_loss_for_log:.{4}f}"
                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # M1 router diagnostics (no TensorBoard needed)
            log_iv = int(getattr(opt, "m1_router_log_interval", 0) or 0)
            if (
                _router_snap is not None
                and log_iv > 0
                and iteration % log_iv == 0
            ):
                dbg_dir = os.path.join(dataset.model_path, "debug")
                os.makedirs(dbg_dir, exist_ok=True)
                csv_path = os.path.join(dbg_dir, "m1_router_usage.csv")
                p_mean, mm, bl = _router_snap
                e0, e1, e2 = float(p_mean[0]), float(p_mean[1]), float(p_mean[2])
                wo, wd, wr, wt = float(mm[0]), float(mm[1]), float(mm[2]), float(mm[3])
                header = "iteration,e0,e1,e2,w_occ,w_doc,w_rank,w_thin,balance_loss\n"
                line = f"{iteration},{e0:.6f},{e1:.6f},{e2:.6f},{wo:.6f},{wd:.6f},{wr:.6f},{wt:.6f},{float(bl):.6f}\n"
                if not os.path.isfile(csv_path):
                    with open(csv_path, "w", encoding="utf-8") as f:
                        f.write(header)
                        f.write(line)
                else:
                    with open(csv_path, "a", encoding="utf-8") as f:
                        f.write(line)

            # Log and save
            training_report(
                tb_writer,
                iteration,
                Ll1_render,
                loss,
                depth_normal_loss,
                ncc_loss,
                occ_loss,
                depth_occ_loss,
                depth_occ_rank_loss,
                occ_thin_loss,
                occ_ng_loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background, kernel_size),
            )
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.05,
                        scene.cameras_extent,
                        size_threshold,
                    )
                    if dataset.disable_filter3D:
                        gaussians.reset_3D_filter()
                    else:
                        gaussians.compute_3D_filter(cameras=trainCameras)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration % 100 == 0 and iteration > opt.densify_until_iter and not dataset.disable_filter3D:
                if iteration < opt.iterations - 100:
                    # don't update in the end of training
                    gaussians.compute_3D_filter(cameras=trainCameras)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                if m1_router_optimizer is not None:
                    m1_router_optimizer.step()
                    m1_router_optimizer.zero_grad(set_to_none=True)

            if soft_stop.should_save_checkpoint(iteration):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )
            if soft_stop.should_exit(iteration):
                # Ensure a checkpoint exists at the stop iteration.
                ckpt_path = scene.model_path + "/chkpnt" + str(iteration) + ".pth"
                if not os.path.isfile(ckpt_path):
                    print("\n[ITER {}] Saving Checkpoint (soft-stop)".format(iteration))
                    torch.save(
                        (gaussians.capture(), iteration),
                        ckpt_path,
                    )
                flag = stop_request_path(scene.model_path)
                if os.path.isfile(flag):
                    try:
                        os.remove(flag)
                    except OSError:
                        pass
                print(
                    f"\n[soft-stop] stopped at checkpoint iter {iteration} "
                    f"(resume with --start_checkpoint {ckpt_path})",
                    flush=True,
                )
                progress_bar.close()
                raise SystemExit(2)


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    normal_loss,
    ncc_loss,
    occ_loss,
    depth_occ_loss,
    depth_occ_rank_loss,
    occ_thin_loss,
    occ_ng_loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/normal_loss", normal_loss.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/ncc_loss", ncc_loss.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/occ_loss", occ_loss.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/depth_occ_loss", depth_occ_loss.item(), iteration)
        tb_writer.add_scalar(
            "train_loss_patches/depth_occ_rank_loss", depth_occ_rank_loss.item(), iteration
        )
        tb_writer.add_scalar("train_loss_patches/occ_thin_loss", occ_thin_loss.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/occ_ng_sharp_loss", occ_ng_loss.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)

    class ValidationConfig(TypedDict):
        name: str
        cameras: Sequence[Camera]

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs: tuple[ValidationConfig, ...] = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {
                "name": "train",
                "cameras": [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    render_result = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_result["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.cuda(), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            config["name"] + "_view_{}/render".format(viewpoint.image_name),
                            image[None],
                            global_step=iteration,
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config["name"] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt_image[None],
                                global_step=iteration,
                            )
                        depth = render_result["median_depth"].squeeze().detach().cpu().numpy()
                        depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
                        depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
                        depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
                        tb_writer.add_images(
                            config["name"] + "_view_{}/depth".format(viewpoint.image_name),
                            depth_color.transpose(2, 0, 1)[None],
                            global_step=iteration,
                        )
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config["name"], l1_test, psnr_test))
                if config["name"] == "test":
                    with open(scene.model_path + "/chkpnt" + str(iteration) + ".txt", "w") as file_object:
                        print(
                            "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config["name"], l1_test, psnr_test),
                            file=file_object,
                        )
                if tb_writer:
                    tb_writer.add_scalar(config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration)
                    tb_writer.add_scalar(config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar("total_points", scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--quiet", action="store_true")
    # Dense by default so soft-stop waits at most ~1k iters for a resume point.
    parser.add_argument(
        "--checkpoint_iterations",
        nargs="+",
        type=int,
        default=list(range(10000, 30001, 1000)),
    )
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    # torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        dataset=lp.extract(args),
        opt=op.extract(args),
        pipe=pp.extract(args),
        testing_iterations=args.test_iterations,
        saving_iterations=args.save_iterations,
        checkpoint_iterations=args.checkpoint_iterations,
        checkpoint=args.start_checkpoint,
        debug_from=args.debug_from,
    )

    # All done
    print("\nTraining complete.")
