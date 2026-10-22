# v20 separate-camera coverage/appearance stress diagnostic

2026-10-21. Reuse immutable v19 query/native binaries and unchanged alpha/order rules.
No training, optimizer, backward or new CUDA build. This tests boundaries of the
prepared cached single-parent query; no GT or geometry gain claim.

Use public7000 flowers/room, v12 fresh cameras _DSC9136/DSCF4755. Baseline precomputed
physical-centroid arrays must literally reproduce all5 v18 fresh baseline hashes.
Freeze source/model/camera/protocol hashes before baseline capture. Then select parents
using baseline geometry only, before any new split output:

- Eligible: native radius>0 and effective opacity>=0.01; cached tile rectangles defined.
- Compute every eligible parent's tile-rectangle area. Let Amax be its maximum.
- Targets are Amax*(1,0.5,0.1,0.01). Sequentially choose the unused ID minimizing absolute
  area-target difference, ties by smaller original ID. Save all eligible IDs,area,rects,
  opacity and the chosen selection before the first split for that model/view.
- Four parents per scene, both held-parent color H and moved physical child-center SH
  color M:16 appearance cases total. Retain every selected parent, including support/
  output degeneracies; do not replace a difficult/slow parent after seeing results.

One original plus4*(one4-child H forward + full H + full M) per scene:26 native forwards.
The local M check may reuse the H geometry cache with M colors; verify all full H/M
noncolor fields literal and child/external geometry independent of appearance. Re-run
v19 all-visible cache gates, all tile-list comparisons, global RGB/alpha/last+median
literal gates and unaffected-region gates for each new case. No error tolerance.
Save baseline inputs, parent/child parameters, colors and local outputs as lossless
baseline-plus-bounding-patch data for CPU reconstruction. Every mismatch is retained.

Record affected pixels/tiles and a conservative static candidate*pixel work count,
including all candidates before early-stop. This metadata is not observed kernel time.
If real max coverage is small, state that this protocol did not create a large-coverage
stress; do not synthetically enlarge triangles and call them real data.

Only after all16 correctness cases pass, freeze a separate balanced timing protocol.
Timing local endpoints must include the native4-child forward for the selected H/M
appearance, not reuse its geometry free from a correctness pass. Report cold/cache
setup, output clones, prepared-event exclusions and cases where local is slower.
No fallback routing or implementation change is permitted in this fixed-code test.
