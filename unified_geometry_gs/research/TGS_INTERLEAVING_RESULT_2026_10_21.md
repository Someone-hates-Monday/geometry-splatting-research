# Interleaving-aware subdivision certificate: valid on frozen inputs, often loose

2026-10-21. Goal remains open: no new training or demonstrated geometry/GT/efficiency gain.
This phase adds mathematical analysis and reproducible fixed-input validation. It does
not turn the negative v16 ray-order result into an optimization success.

## What changed

TGS_INTERLEAVING_CERTIFICATE_NOTE_V17.md derives an exact, division-free RGB decomposition
for same-color children interleaved with otherwise unchanged external layers. It combines
the known fixed-slot window replacement with an external-layer sum weighted by the change
in the child transmittance before each external layer. Prefix scans give O(N+children)
evaluation for a single recorded pixel. This handles arbitrary child/external interleaving
under the stated assumptions, including alpha0/1 in exact arithmetic.

The native bound additionally includes the exact omitted tails from skip-current stopping
and a conservative conditional FP32 rounding envelope. It treats the recorded CUDA alpha
values as exact inputs; it does not certify preprocessing, __powf, SH or geometric error.
Common color and unchanged external effective layers are verified per sampled pair.
Normal-range, nonnegative-color and no-overflow premises are checked. The instruction
model is checked on recorded traces; CUDA compiler semantics in general remain unproved.

The familiar fixed-slot deletion/replacement principle is prior art, including
[GaussianPOP §3.2](https://arxiv.org/html/2602.06830). Render-once local evaluation is
also already used there. [SteepGS](https://vita-group.github.io/SteepGS/) addresses
splitting through a second-order loss model. No novelty of our rearranged identity or
method is established by the targeted search or by this experiment.

## Exact development checks

SymPy1.14 checks18 placements: two child layers interleaved with two external layers,
for all three original parent slots. A written general proof accompanies those checks;
enumerating18 cases alone would not prove the general statement.

All three previously used v15 room pixels satisfy the decomposition as exact rational
identities from recorded FP32 values. Their native max-channel changes are bounded with
bound/error ratios1.02344,1.22231,1.31727. They are development examples, not confirmation.
The original three all lack early stopping.

A separate software example with two capped alpha0.99 layers verifies that FP32
skip-current can omit a tail of about0.00989999, much larger than nominal threshold1e-4.
This illustrates an omitted-current error; it is not a real-scene measurement.

## Frozen real-input validation

Two existing public checkpoints, flowers/room7000. Existing v14 cameras0/1; event1/2
exclude the old development events0/5. Eight pixels/case,64 paired endpoints in total.
Four points use each old change rectangle when nonempty and the remaining positions
use deterministic global SHA256 coordinates. Empty rectangles retain global positions.
Because rectangles and views have been seen before, this is directed arithmetic/mechanism
validation, not independent photographic confirmation or a generalization experiment.
All coordinates and hashes were frozen before any new GPU output.

- Exactly12 native full forwards:4 original views and8 held-parent-color splits.
- All five full-image output arrays literally match the sealed v14 hashes.
-256 observer calls; all128 RGB/alpha/last-ordinal endpoints literally match native.
-64/64 exact rational interleaving identities and channelwise native error containments
  pass, including20 paired pixels with native early stopping.
-9,687 exact RN transmittance updates pass. All29,061 accepted channel updates are
  compatible with the recorded FP32 first product and RN fused accumulation;28,551
  are also compatible with separate multiplication/addition.
-Six pairs have active order crossings.44 use an explicit zero-alpha parent because
  the original parent is absent from that tile; its arbitrary position0 is fixed in
  the pre-run protocol and changes no exact composite.
-No premise failures were discarded. Both models/cameras remain unchanged; no optimizer,
  backward, training, GT metric comparison or latency measurement is performed.

## Tightness determines the next action

All64 errors are contained, but13 are nonzero and51 are exactly zero. Across the13
nonzero errors, bound/error minimum1.01914, median5.26810, maximum50.69837. The maximum
absolute bound is0.000685528; maximum nonzero observed change is0.000226125.
All51 zero-error points still receive positive generic bounds, ranging from
4.92852e-7 to0.000293089. They are retained, not removed to improve statistics.

Examples of looseness: room test event2 pixel(605,252) has actual change3.27826e-6
and bound1.66202e-4; room train event2 pixel(234,769) has1.70618e-5 versus6.85528e-4.
Both stop early. Summing the before/after omitted tails loses their cancellation,
and separate endpoint rounding envelopes ignore shared arithmetic. Order-term signed
cancellation can also loosen the triangle inequality even without stopping.

Thus the current formula is a conditional analysis tool, not a validated split rule.
Eight sampled pixels cannot certify a whole image or a training trajectory. Applying
this per-pixel certificate to a global split decision would require coverage of all
relevant pixels/views and a measured affordable implementation. No speed, global safety,
training stability, GT improvement or novelty claim follows from these checks.

Next: derive coupled prefix/tail cancellation and shared-rounding bounds before gathering
new evidence. Use a separately frozen cohort to test whether the tighter bound is useful,
including inactive/zero and early-stop cases. If it remains mostly non-discriminative or
requires work comparable to a full rerender for every split, abandon it as a practical
decision rule rather than presenting mathematical validity as method performance.

## Reproduction and retained failure

Sources: tgs_interleaving_certificate_v17.py, audit_tgs_interleaving_development_v17.py,
trace_tgs_interleaving_validation_v17.py, audit_tgs_interleaving_validation_v17_v2.py.
Outputs: experiments/tgs_interleaving_v17; figure bound_coverage.{png,pdf}.
Development audit, GPU recapture and final CPU audit all terminate with exit code0.

Initial CPU audit exited1 because it demanded every trace intermediate be finite;
outside-support max distance legitimately remains -Infinity. Original script, audit
plan and failure explanation remain. The derivativev2 checks finite IDs/alpha/flags/T/RGB
and keeps those legitimate intermediate sentinels. No GPU rerun or bound/tolerance change.
The open median_depth_note.tex remains untouched.
