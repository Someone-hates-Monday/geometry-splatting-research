# A — Field-intrinsic occupancy on \(T=1/2\)

## Idea

G-GS defines the surface as \(t_{\mathrm{med}}=T^{-1}(1/2)\). Screen \(L_{\mathrm{occ}}\) penalizes \(\alpha\approx\tfrac12\), which is a different quantity. Method **A** penalizes a **shallow transmittance drop** at the final median-depth search bracket:

\[
s=\frac{|T(d_{\mathrm{lo}})-T(d_{\mathrm{hi}})|}{d_{\mathrm{hi}}-d_{\mathrm{lo}}},
\qquad
L_{\mathrm{occ}}^{\mathrm{field}}=\mathbb{E}_{\mathcal{V}}\big[\exp(-\beta\,\hat s)\big],
\quad
\hat s=s/\mathrm{sg}(\mathrm{median}\,s).
\]

## Implementation notes

- Rasterizer `out_mdepth` is now **2 channels**: `[0]=median_depth`, `[1]=interface_slope`.
- Forward slope is the exact bracket finite difference.
- Backward uses a **straight-through** path through analytical \(|\partial T/\partial t|\) (same vacancy terms as median-depth grads). Document this when reporting results.
- Rebuild required: `pip install ./submodules/diff-gaussian-rasterization --no-build-isolation`.

## Run (vs M1-3)

```bash
# A only (no screen occ/doc)
./scripts/run_m1_field_occ.sh "$UGGS_DTU_ROOT/scan24" ./output_compare/m1_field_occ/scan24
./scripts/run_m1_field_occ.sh "$UGGS_DTU_ROOT/scan105" ./output_compare/m1_field_occ/scan105

# M1-3 control
./scripts/run_m1.sh "$UGGS_DTU_ROOT/scan24" ./output_compare/m1_ablate_03_full/scan24
```

Compare with `scripts/compare_models.py` against the same `ggs_baseline`.

## Pass / fail

Average Chamfer overall over scan24+scan105 must improve on M1-3 by at least \(5\times10^{-4}\), without mean s2d getting worse. If not, stop; do not add platform length or Doc-field yet.
