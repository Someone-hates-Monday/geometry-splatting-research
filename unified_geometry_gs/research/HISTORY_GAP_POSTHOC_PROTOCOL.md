# Read-only mechanism check after the failed fixed pilot

2026-10-16 Asia/Shanghai. The full history_gap_pilot_v1 completed and failed its
predefined geometry gate; independent audit confirms the failure. This is a
post-hoc diagnostic, not preregistered evidence or a replacement quality gate.
No new training, weight search, altered renderer or changed evaluation.

Question: on identical current camera/pixel queries, did the learned candidate
reduce the implemented loss on nonempty rays, and how much of the full loss is
the analytic empty-ray constant? This distinguishes failure to optimize the
implemented objective from failure of that objective to improve GT. It cannot
identify a causal failure mechanism from two fitted models.

Use both actual final30000 PLYs, saved filters, original private pinned query,
CSR and warp loss binary. For every one of the42 train cameras, uniformly sample
512 pixels with a separate Generator seed[13024,numeric_camera]. Same indices
for both models. All no-grad; no test RGB/GT used and no model write. Existing
empty contributors remain loss(log4)^2, gradient0, full512 denominator.
Save per-sampled-pixel values, active/empty indicator, contributor count and
native info[8] (status,t,D,Omega,iterations,n,left,right). Code2 is a tail
calculation and is NOT labelled an absent-root certificate. Info is Float64
engineering output, not new interval/MP certification.

Report both full-sample means, nonempty conditional means, paired-common-active
means, empty transitions, native status counts, and exact additive decomposition
full_mean = empty_fraction*(log4)^2 + sum_nonempty_loss/full_sample_count.
All42 cameras included; no favorable view selection or significance claim.
If loss decreases while GT does not, report objective/task mismatch at this
fixed exploratory pair. If it does not decrease, retain that result too; do
not automatically change lambda, mask, sampling or the model in response.
