# Warp-per-ray kernel: fixed correctness and cost comparison

2026-10-15 before warp results. Original serial source/binary and graphV2
remain frozen. Replace only numerical reduction/work distribution: one
32-lane warp per ray, striding through all nodes, compensated local sums
and double-double prefix products, fixed warp reduction tree. Mathematical
loss, early exits, independent interior bisection, caps, side-density seeds
and raw graph are unchanged. Float64 signs remain numerical, not certified.

All original891 real rays/51524 nodes/9 anomalies/3zero losses plus existing
64 synthetic models and two absent updates must pass the original whole
ray1e-5/block1e-4 gradient and loss absolute1e-9 gates versus independent
MP references. Retain every failure. Repeat10 fixed coefficient cases,
invalidrho0 rejection, and add explicit513/1025-node coefficient rays with
last primitive necessary for a finite root, to expose truncation.
No fallback to serial may be hidden in warp timings or correctness counts.

Load serial/warp binaries into one process. Two initial warmups per arm,
then8 measurements alternating order. Record kernel loss/seeds and full
geometry+loss+raw backward event costs, allocation included, setup/transfer/
compile separately. Same data, device, graph, stream, no overlapping jobs.
Report all observations and medians; they are not statistical confidence.
Adopt for the next adapter only if all gates pass and full warm median
warp/serial<=.95. Otherwise preserve the failed implementation and diagnose.
This is a local GPU loss cost comparison, not full-training acceleration,
newness, photometric or GT evidence. No training begins in this stage.
