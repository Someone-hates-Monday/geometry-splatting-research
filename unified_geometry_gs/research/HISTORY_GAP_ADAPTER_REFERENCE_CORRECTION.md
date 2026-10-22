# V2 oracle-selection bug; original failure retained

V2 passed original image replay, every captured source/raw/ID byte check
and the amended parameter/gradient checks, then failed scatter comparison.
The runner incorrectly read the world_root_vjp_v1 reference, which is a
median-depth gradient, as the squared-history-loss gradient. The correct
independent MP loss adjoint is history_gap_raw_v1/reference (same pixels,
sections and primitive IDs). This is an oracle-selection error, not a
numerical verdict on the adapter. V2 failed state/source ZIP remain.

V3 keeps the adapter/kernel and all gates unchanged, verifies hashes of
both input and loss-reference arrays and their exact mapping, and stores
scatter arrays/metrics before any gate assertion. It also persists step
history after every update. No optimizer update occurred in V1 or V2.
