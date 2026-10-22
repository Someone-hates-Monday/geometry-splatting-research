# Setup: merge G-GS into this subproject

Current contents of **unified_geometry_gs** (ours only):

- `configs/` — paths.yaml, m1/m2/m3.yaml
- `losses/` — unified_occupancy.py, weighted_geometry.py (stubs)
- `docs/` — MVP_milestones.md
- `scripts/` — run_baseline.sh, run_m1/m2/m3.sh
- `README.md`, `SETUP.md`

**Steps to get a full working tree (G-GS + ours):**

1. **Back up the above** (e.g. copy `unified_geometry_gs` to `unified_geometry_gs_backup`).

2. **Clone G-GS into a temporary directory** (from `SuGaR-main`):
   ```bash
   git clone --recursive https://github.com/HKUST-SAIL/Geometry-Grounded-Gaussian-Splatting.git gggs_temp
   ```

3. **Move G-GS contents into unified_geometry_gs** (so both codebases live in one folder):
   - Copy everything from `gggs_temp/` into `unified_geometry_gs/` (train.py, gaussian_renderer/, scene/, arguments/, utils/, submodules/, requirements.txt, etc.).
   - Do **not** overwrite our `configs/`, `losses/`, `docs/`, `scripts/`, or README if you prefer to keep our versions; merge by hand if needed.

4. **Remove the temp clone**: `rm -rf gggs_temp`

5. **Install deps** (see G-GS README): submodules, conda env, etc.

After this, `unified_geometry_gs` contains both G-GS and our additions and remains independent from SuGaR.
