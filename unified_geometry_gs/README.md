# Unified Geometry GS — Independent Subproject

> **接手入口（2026-10-22）**：[项目交接](../HANDOFF.md)。当前目录已有 G-GS 与科研改动；下面早期初始化段落是历史记录，**不要对现有目录执行 `rm -rf` 或重新克隆覆盖**。最新封存研究为 TGS v29，科研总目标尚未完成。

This folder is a **self-contained subproject**: unified depth–occupancy + confidence/semantic weighting on top of [Geometry-Grounded Gaussian Splatting (G-GS)](https://github.com/HKUST-SAIL/Geometry-Grounded-Gaussian-Splatting). It does not import any SuGaR code. You can share data/output with the parent repo via `configs/paths.yaml`.

---

## 1. First-time setup: clone G-GS into this directory

From the **parent repo root** (e.g. `SuGaR-main`), run:

```bash
# Option A: clone into a temp dir, then merge into this folder
git clone --recursive https://github.com/HKUST-SAIL/Geometry-Grounded-Gaussian-Splatting.git gggs_base
# Then copy all files from gggs_base into unified_geometry_gs (this folder):
#   - train.py, mesh_extract.py, mesh_extract_tetrahedra.py, render.py, evaluate_dtu_mesh.py, metric.py
#   - gaussian_renderer/, scene/, arguments/, utils/, submodules/, assets/, scripts (G-GS), etc.
# So that this folder contains both G-GS and the additions below (configs/, losses/, docs/, scripts/).
```

**Option B (simplest):** If this folder is still empty of G-GS, you can clone directly on top (this will replace the folder content with G-GS; then re-add the contents of `configs/`, `losses/`, `docs/`, `scripts/` from git or backup):

```bash
# From SuGaR-main:
rm -rf unified_geometry_gs   # or move configs, losses, docs, scripts out first
git clone --recursive https://github.com/HKUST-SAIL/Geometry-Grounded-Gaussian-Splatting.git unified_geometry_gs
cd unified_geometry_gs
# Then copy back: configs/, losses/, docs/, scripts/, and this README from your backup.
```

**Recommended:** Keep a copy of `configs/`, `losses/`, `docs/`, `scripts/` and this `README.md` before cloning. After cloning G-GS into `unified_geometry_gs`, paste these back so the final layout is:

- G-GS: `train.py`, `gaussian_renderer/`, `scene/`, `arguments/`, `utils/`, `submodules/`, ...
- Ours: `configs/`, `losses/`, `docs/`, `scripts/` (and this README).

---

## 2. Environment

Create a dedicated env (recommended). **PyTorch 的 CUDA 版本必须和系统 CUDA 一致**（如系统 11.8 → 用 `cu118`），否则 submodules 编译会报版本不匹配。详见 **INSTALL.md**。

```bash
conda create -n uggs python=3.12
conda activate uggs
# 系统 CUDA 11.8 用 cu118，CUDA 12 用 cu121：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install ninja
pip install submodules/diff-gaussian-rasterization --no-build-isolation
pip install submodules/warp-patch-ncc --no-build-isolation
pip install submodules/simple-knn --no-build-isolation
pip install "git+https://github.com/rahul-goel/fused-ssim/" --no-build-isolation
```

---

## 3. Shared paths (optional)

Edit `configs/paths.yaml` to point `data_root` and `output_root` at the parent repo’s data/output so both SuGaR and this subproject use the same folders. Outputs go to `output_root/uggs/` (e.g. `../output/uggs/baseline/`, `../output/uggs/m1/`).

---

## 4. Milestones and scripts

- **M0 (baseline):** `scripts/run_baseline.sh <path_to_dtu> <output_dir>`
- **G-GS leave-one-out / MV-split (no DN, no MV, NCC-only, geo-only vs full baseline):** `GGS_VARIANTS=ncc_only,geo_only ./scripts/run_ggs_ablation.sh scan24 scan105` — summarize: `python scripts/summarize_m1_ablation.py --tag_prefix ggs_ablate_`
- **Phase-2 NCC-core / corr-weight (no DN, no geo):** `./scripts/run_ggs_phase2_ncc.sh scan24 scan105`
- **NCC-gated depth-normal (complementary DN):** `./scripts/run_ggs_ncc_gate_dn.sh scan24 scan105 scan122` — compare vs `ggs_ablate_no_dn`
- **NCC fail coverage (on no_dn recipe):** `./scripts/run_ggs_ncc_fail.sh scan24 scan105 scan122` — `--lambda_ncc_fail 0.15`
- **2-nearest trusted-band NCC (on no_dn):** `./scripts/run_ggs_mv2.sh scan24 scan105 scan122` — `--multi_view_ncc_views 2`
- **d2s tail diagnose:** `python scripts/diagnose_d2s.py --a <aligned_a.ply> --b <aligned_b.ply> --scan 122 --labels baseline no_dn`
- **Mesh extract sweep (frozen no_dn Gaussians):** `./scripts/run_ggs_extract_sweep.sh scan24 scan105 scan122` — voxel / depth_max / trunc / clusters
- **Adopted extract default:** `--trunc_voxel_multiplier 4` (`ex_trunc4`; also strong: `--voxel_size 0.001`)
- **Research protocol P1+P2 (σ_z + t\* grid, no train):** `./scripts/run_research_p1_p2.sh` — see `docs/RESEARCH_PROTOCOL.md`
- **NCC trusted-band threshold (P3):** `--ncc_mask_thresh 0.9` (default; sweep 0.3/0.5/1.2)
- **M1 (depth–occupancy):** `scripts/run_m1.sh` (L_occ + L_depth_occ in `train.py`)
- **A (field occ on \(T=1/2\) slope):** `scripts/run_m1_field_occ.sh` — see `docs/FIELD_OCC_A.md` (requires rebuilt rasterizer with 2-channel mdepth)
- **M1 structured (occlusion ranking at α-boundaries):** `scripts/run_m1_structured.sh`, or `--lambda_depth_occ_rank` with `--occ_rank_margin` / `--occ_rank_alpha_soft`
- **M1 thin-boundary (thin wall / concave helper):** `scripts/run_m1_thin.sh`, or `--lambda_occ_thin` with `--occ_thin_edge_power`
- **M1 geometry-aware soft router (dynamic M1 weights):** `scripts/run_m1_router.sh` — see `docs/M1_GEOMETRY_ROUTER.md`
- **M1 ablation (M1-1…M1-4 + metrics table):** `scripts/run_m1_ablation.sh` or `scripts/run_m1_ablation.ps1` — see `docs/EXPERIMENT_LOG_M1.md` §4; summary: `python scripts/summarize_m1_ablation.py`
- **M1 mesh-focused (tune for DTU Chamfer):** `scripts/run_m1_mesh_focus.sh` — see `docs/OPTIMIZATION_EXPERIMENTS.md`
- **M1 mesh moderate (softer than mesh_focus):** `scripts/run_m1_mesh_mid.sh`
- **M2 (weighting):** `scripts/run_m2.sh` (after wiring w_geo, w_sem)
- **M2-w_geo (implemented):** `scripts/run_m2.sh` enables `--use_w_geo`; default is **ncc-only weighting** (`--w_geo_ncc_only`) for better stability.
- **M3 (mesh):** `scripts/run_m3.sh <output_dir>`

See `docs/MVP_milestones.md` for tables and success criteria.

Further tuning (M1 hyperparameters, presets): **`docs/OPTIMIZATION_EXPERIMENTS.md`**.

External / SuGaR comparison pitfalls (fairness checklist): **`docs/COMPARISON_EXTERNAL.md`**.  
Batch compare several UGGS runs vs same baseline: **`scripts/compare_scan_methods.sh`**.
