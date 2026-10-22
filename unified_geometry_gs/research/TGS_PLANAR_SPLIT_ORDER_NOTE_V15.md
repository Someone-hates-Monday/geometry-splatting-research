# Planar midpoint refinement: physical ray depth versus center-sort key

2026-10-21. Standard affine/barycentric and alpha-compositing identities, applied to the
observed TGS mechanism. No general novelty, GT improvement or new sorting algorithm claim.

Let a,b,c be a nondegenerate planar triangle, g=(a+b+c)/3. Exact world midpoint four-way
refinement gives three corner centroids g_i=g+(v_i-g)/2, and central centroid g_4=g.
Every child lies on the parent plane. Thus for a camera ray that intersects a child,
its intersection with the parent's infinite plane is the SAME point/depth, assuming
nonparallel ray and a unique finite intersection. Refinement itself does not change
the plane's physical ray depth at that pixel.

However, for the affine camera-view depth z(v), the center keys become

    z(g_i)=z(g)+(z(v_i)-z(g))/2,  i=1,2,3,
    z(g_4)=z(g).

Consequently a child's center-sort key can cross a third primitive's center key without
any change in either physical plane. A sufficient strict no-crossing condition with
all external center keys z_j is

    min_j |z_j-z(g)| > (1/2) max_i |z(v_i)-z(g)|.

If this holds, every child has the same side of every external key as its parent, so
no external center ordering crosses. Equality/ties require separate handling, and the
condition is sufficient, not necessary. It does not resolve wrong center ordering
already present, per-pixel culling, tile membership, numerical rounding or later motion.
The observed native implementation uses center-depth bits in the tile key; the relevant
pinned source is cuda_rasterizer/rasterizer_impl.cu::duplicateWithKeys.

For two active layers with alpha a,b, colors c,d and unchanged suffix/prefix, swapping
their order changes the local RGB by a*b*(c-d), multiplied by incoming transmittance.
This follows directly from [a*c+(1-a)*b*d]−[b*d+(1-b)*a*c]. Opacity is unchanged
in real arithmetic, because (1-a)(1-b) is commutative. Thus small isolated alpha
difference does not bound RGB change when the relative order changes. Pixel sorting
methods already address this general issue; see TGS_WINDOW_PRIOR_WORK_V15.md.

Native FP32 midpoint coordinates need not lie exactly on the same real plane; the
identity above is an exact-arithmetic statement. v15 recapture/replay isolates order
on actual cached inputs, while any physical-depth method must account for numerical
conditions separately. Moving children back to the parent slot is a mechanism probe,
not a proof of correct physical ordering or an optimization strategy.
