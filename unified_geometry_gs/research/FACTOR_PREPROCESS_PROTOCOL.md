# Direct-factor FP64 candidate: replacement protocol before quality measurement

2026-10-12. This is a documented replacement of the unstarted corrected64
candidate, not a rewrite of its original protocol or a successful quality result.
Original corrected64 passed 72 regular real and 48 synthetic primitives but
failed the subsequently fixed, pre-quality 44-case extreme-axis derivative gate.
All inputs, failures, source archives and the superseded queue are retained.

The new factor64 intervention combines the three local chain-rule corrections
with a direct precision-factor VJP. It avoids the covariance inverse round trip
which assumes an exactly orthogonal rotation from a float32-normalized quaternion.
The physical float64 reference remains the admission reference: relative L2 error
at most .001, with denominator floor 1e-5, including structural zero groups.
Every saved 44-case stress input and all 120 regular real/synthetic inputs and
upstreams must pass without exclusions. This does not certify the full renderer,
hard masks, or all forward values: four extreme forward components failed their
original gate and remain recorded; factor64 does not change forward rendering.

Control is the already completed legacy64 branch in
`corrected_quality55_v1/legacy64`. It was selected and trained before any new
mesh/image quality measurement, and is retained regardless of the candidate's
outcome. Its actual 30k model and integration hashes are saved before candidate
training. The stopped old parent was retired only after its control child finished;
corrected64 never started. See `protocol_deviation_factor_v1.json`.

Both arms retain local FP64 world-backward intermediates, original float32 outputs
and contributor accumulation, five root subdivisions, query batch size two,
original NCC and loss weights. factor64 additionally promotes the new inverse
helper, so this combined pilot does not isolate each correction or arithmetic
change. Source identity between the probe and full backend is checked for all
three local CUDA functions; forward/configuration source bytes remain unchanged.

Use the same verified iteration-6999 checkpoint, sidecar, appearance optimizer
binding, RNG, camera queue and complete initial-state digest. Run a new candidate
20-update smoke, checking finite loss/gradients and identical first camera and
first forward loss components against the retained control. Then continue a fresh
candidate from the same iteration-6999 state to iteration 30000. The smoke is not
resumed as the quality branch. CUDA trajectories need not be deterministic.

Keep the original quality gate: mean overall over evaluator seeds 0/1/2 must
improve by at least 1%; mean s2d must not increase; 49 TRAIN-view mean PSNR must
not fall by more than .2 dB. Extract both arms at voxel .002, truncation multiplier
4, depth max 8, cluster one, then identical alignment and evaluation. Evaluator
seeds are repetitions of mesh sampling, not independent training replications.
There is no held-out image or speed claim. Sequential wall time is accounting.
Do not choose a different control, remove failures or change thresholds afterward.

A passing pilot warrants separately justified ablations, independent training
repeats, other scenes, held-out views and novelty review. A failed pilot is saved
as a negative result rather than automatically enlarged. Every queued stage
guards actual source hashes and records start, exit code, outputs or failure.

Workflow reference: Kassis et al. (2026),
[Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065).
